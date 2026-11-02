#!/usr/bin/env python3
"""Compare every exported HF tensor to the original CPU-loaded DCP weights.

Supports only the explicitly inspected Qwen3-MoE checkpoint layout. Unknown
tensor keys, shape/dtype mismatches or missing/extra HF weights fail closed.
No source checkpoint or installed library is modified; no NPU is initialized.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch


LAYER = "decoder.layers."
KNOWN = {
    "embedding.word_embeddings.weight", "output_layer.weight", "decoder.final_layernorm.weight",
    LAYER+"self_attention.linear_proj.weight", LAYER+"self_attention.linear_qkv.layer_norm_weight",
    LAYER+"self_attention.linear_qkv.weight", LAYER+"self_attention.q_layernorm.weight",
    LAYER+"self_attention.k_layernorm.weight", LAYER+"pre_mlp_layernorm.weight",
    LAYER+"mlp.router.weight", LAYER+"mlp.experts.experts.linear_fc1.weight",
    LAYER+"mlp.experts.experts.linear_fc2.weight",
}


def expected_weights(state, config):
    if set(state) != KNOWN:
        raise ValueError(f"unrecognized DCP tensor layout: missing={KNOWN-set(state)} extra={set(state)-KNOWN}")
    yield "model.embed_tokens.weight", state["embedding.word_embeddings.weight"]
    yield "lm_head.weight", state["output_layer.weight"]
    yield "model.norm.weight", state["decoder.final_layernorm.weight"]
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    dim = config["head_dim"]
    if heads % kv_heads: raise ValueError("unsupported grouped QKV layout")
    per_group = heads // kv_heads
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        for source, target in (
            ("self_attention.linear_proj.weight", "self_attn.o_proj.weight"),
            ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
            ("self_attention.q_layernorm.weight", "self_attn.q_norm.weight"),
            ("self_attention.k_layernorm.weight", "self_attn.k_norm.weight"),
            ("pre_mlp_layernorm.weight", "post_attention_layernorm.weight"),
            ("mlp.router.weight", "mlp.gate.weight"),
        ):
            yield prefix+target, state[LAYER+source][layer]
        qkv = state[LAYER+"self_attention.linear_qkv.weight"][layer].reshape(
            kv_heads, per_group+2, dim, config["hidden_size"])
        for name, tensor in (("q", qkv[:, :per_group]), ("k", qkv[:, per_group]), ("v", qkv[:, per_group+1])):
            yield prefix+f"self_attn.{name}_proj.weight", tensor.reshape(-1, config["hidden_size"])
        fc1 = state[LAYER+"mlp.experts.experts.linear_fc1.weight"][layer]
        fc2 = state[LAYER+"mlp.experts.experts.linear_fc2.weight"][layer]
        for expert in range(config["num_experts"]):
            gate, up = fc1[expert].chunk(2, dim=0)
            eprefix = prefix+f"mlp.experts.{expert}."
            yield eprefix+"gate_proj.weight", gate
            yield eprefix+"up_proj.weight", up
            yield eprefix+"down_proj.weight", fc2[expert]


def main():
    from safetensors import safe_open
    from torch.distributed.checkpoint import FileSystemReader, load
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="original actor/dist_ckpt")
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.checkpoint.resolve()):
        raise ValueError("refusing overwrite/source mutation")
    config_path = args.hf_model/"config.json"
    config = json.loads(config_path.read_text())
    if config["model_type"] != "qwen3_moe": raise ValueError("only inspected Qwen3MoE layout supported")
    index_path = args.hf_model/"model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    reader = FileSystemReader(args.checkpoint)
    meta = reader.read_metadata()
    tensor_meta = {key: val for key, val in meta.state_dict_metadata.items() if isinstance(val, TensorStorageMetadata)}
    if set(tensor_meta) != KNOWN: raise ValueError("checkpoint contains unknown weight tensors")
    print("loading original DCP weight tensors into CPU memory", flush=True)
    state = {key: torch.empty(val.size, dtype=val.properties.dtype, device="cpu") for key, val in tensor_meta.items()}
    load(state, storage_reader=reader)
    handles = {name: safe_open(str(args.hf_model/name), framework="pt", device="cpu") for name in sorted(set(index.values()))}
    checked, total_elements = set(), 0
    digest = hashlib.sha256()
    for key, original in expected_weights(state, config):
        if key not in index: raise ValueError(f"missing HF tensor {key}")
        actual = handles[index[key]].get_tensor(key)
        if actual.shape != original.shape or actual.dtype != original.dtype or not torch.equal(actual, original):
            raise ValueError(f"HF weight mismatch: {key}")
        checked.add(key); total_elements += actual.numel()
        digest.update(key.encode()+b"\0")
        digest.update(actual.contiguous().view(torch.uint8).numpy().tobytes())
    if checked != set(index): raise ValueError(f"unexpected HF tensors:{set(index)-checked}")
    source_files = sorted(args.checkpoint.glob("*.distcp"))
    paths = [args.checkpoint/".metadata", args.checkpoint/"metadata.json", config_path, index_path, Path(__file__)]
    def sha(path):
        h=hashlib.sha256()
        with path.open("rb") as f:
            for b in iter(lambda:f.read(8<<20),b""):h.update(b)
        return h.hexdigest()
    payload = {"status":"ALL_EXPORTED_TENSORS_EQUAL_ORIGINAL_DCP", "tensors_checked":len(checked),
               "elements_checked":total_elements,"canonical_hf_tensor_sha256":digest.hexdigest(),
               "protocol":{"cpu_only":True,"checkpoint_readonly":True,"full_tensor_comparison":True},
               "input_sha256":{str(p):sha(p) for p in paths},
               "source_shard_sha256":{str(p):sha(p) for p in source_files},
               "export_shard_sha256":{str(args.hf_model/name):sha(args.hf_model/name) for name in sorted(handles)},
               "scope":"Full CPU weight equality including QKV and gate/up conversion. Does not certify EP8/EP16 inference-token parity or equal-time runtime behavior."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({key:payload[key] for key in ("status","tensors_checked","elements_checked")},sort_keys=True))


if __name__=="__main__":main()
