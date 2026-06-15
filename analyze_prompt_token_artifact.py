#!/usr/bin/env python3
"""Inspect compact prompt-token-layer MoE artifacts and compare epochs."""

import argparse
import json
import csv
from pathlib import Path
from typing import Any

import torch


def _load_manifest(epoch_dir: Path) -> dict[str, Any]:
    manifest_path = epoch_dir / "manifest.pt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return torch.load(manifest_path, map_location="cpu")


def _resolve_epoch_dir(artifact_root: Path, epoch: int) -> Path:
    epoch_dir = artifact_root / f"epoch_{int(epoch):04d}"
    if not epoch_dir.exists():
        raise FileNotFoundError(f"Epoch artifact directory not found: {epoch_dir}")
    return epoch_dir


def _find_prompt_entry(manifest: dict[str, Any],
                       *,
                       prompt_hash: str | None,
                       prompt_index: int | None) -> dict[str, Any]:
    prompt_files = manifest.get("prompt_files", [])
    if prompt_hash is None and prompt_index is None:
        raise ValueError("Either prompt_hash or prompt_index must be provided.")
    if prompt_hash is not None:
        for entry in prompt_files:
            if entry.get("prompt_hash") == prompt_hash:
                return entry
        raise KeyError(f"Prompt hash not found in manifest: {prompt_hash}")
    assert prompt_index is not None
    if prompt_index < 0 or prompt_index >= len(prompt_files):
        raise IndexError(
            f"prompt_index={prompt_index} out of range [0, {len(prompt_files) - 1}]")
    return prompt_files[prompt_index]


def _load_prompt_payload(epoch_dir: Path, entry: dict[str, Any]) -> dict[str, Any]:
    file_path = epoch_dir / entry["file"]
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt artifact not found: {file_path}")
    return torch.load(file_path, map_location="cpu")


def _top_experts_from_counts(counts: torch.Tensor,
                             top_n: int) -> list[dict[str, Any]]:
    flat = counts.to(dtype=torch.int64).flatten()
    nonzero = torch.nonzero(flat > 0, as_tuple=False).flatten().tolist()
    ranked = sorted(nonzero,
                    key=lambda idx: (-int(flat[idx].item()), int(idx)))
    return [
        {
            "expert_id": int(expert_id),
            "count": int(flat[expert_id].item()),
        }
        for expert_id in ranked[:max(int(top_n), 0)]
    ]


def _selected_set(counts: torch.Tensor) -> set[int]:
    flat = counts.to(dtype=torch.int64).flatten()
    return {
        int(idx) for idx in torch.nonzero(flat > 0, as_tuple=False).flatten().tolist()
    }


def _cosine_similarity(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs = lhs.to(dtype=torch.float32).flatten()
    rhs = rhs.to(dtype=torch.float32).flatten()
    lhs_norm = float(torch.linalg.vector_norm(lhs).item())
    rhs_norm = float(torch.linalg.vector_norm(rhs).item())
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return 0.0
    return float(torch.dot(lhs, rhs).item() / (lhs_norm * rhs_norm))


def _jaccard(lhs: set[int], rhs: set[int]) -> float:
    if not lhs and not rhs:
        return 1.0
    union = lhs | rhs
    if not union:
        return 0.0
    return float(len(lhs & rhs)) / float(len(union))


def _safe_slice(payload: dict[str, Any], token_position: int,
                layer_idx: int) -> tuple[torch.Tensor, int]:
    counts = payload["counts"]
    route_rows = payload["route_rows"]
    if token_position < 0 or token_position >= counts.shape[0]:
        return torch.zeros(counts.shape[-1], dtype=counts.dtype), 0
    if layer_idx < 0 or layer_idx >= counts.shape[1]:
        return torch.zeros(counts.shape[-1], dtype=counts.dtype), 0
    return counts[token_position, layer_idx], int(route_rows[token_position,
                                                            layer_idx].item())


def _list_prompts(manifest: dict[str, Any], *, limit: int | None) -> None:
    prompt_files = manifest.get("prompt_files", [])
    print(
        f"epoch={manifest.get('epoch')} prompts={manifest.get('num_prompts')} "
        f"num_experts={manifest.get('num_experts')} max_num_layers={manifest.get('max_num_layers')}")
    max_rows = len(prompt_files) if limit is None else min(len(prompt_files),
                                                           int(limit))
    for idx, entry in enumerate(prompt_files[:max_rows]):
        print(
            f"[{idx}] prompt_hash={entry['prompt_hash']} "
            f"max_token_position={entry['max_token_position']} "
            f"num_layers={entry['num_layers']} file={entry['file']}")


def _emit(data: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(
        f"prompt_hash={data['prompt_hash']} token_position={data['token_position']} "
        f"layer_idx={data['layer_idx']}")
    epoch_a = data["epoch_a"]
    print(
        f"epoch_a={epoch_a['epoch']} route_rows={epoch_a['route_rows']} "
        f"selected_experts={epoch_a['selected_expert_count']} "
        f"top_experts={epoch_a['top_experts']}")
    if "epoch_b" in data:
        epoch_b = data["epoch_b"]
        print(
            f"epoch_b={epoch_b['epoch']} route_rows={epoch_b['route_rows']} "
            f"selected_experts={epoch_b['selected_expert_count']} "
            f"top_experts={epoch_b['top_experts']}")
        metrics = data["similarity"]
        print(
            "similarity: "
            f"cosine={metrics['cosine_similarity']:.6f} "
            f"jaccard={metrics['jaccard_selected_experts']:.6f} "
            f"shared={metrics['shared_selected_expert_count']} "
            f"prev_only={metrics['prev_only_expert_count']} "
            f"curr_only={metrics['curr_only_expert_count']} "
            f"top1_match={metrics['top1_match']}")


def _build_similarity_row(*,
                          prompt_hash: str,
                          epoch_a: int,
                          epoch_b: int,
                          token_position: int,
                          layer_idx: int,
                          counts_a: torch.Tensor,
                          counts_b: torch.Tensor,
                          route_rows_a: int,
                          route_rows_b: int) -> dict[str, Any]:
    selected_a = _selected_set(counts_a)
    selected_b = _selected_set(counts_b)
    top_a_list = _top_experts_from_counts(counts_a, 1)
    top_b_list = _top_experts_from_counts(counts_b, 1)
    top_a = top_a_list[0]["expert_id"] if top_a_list else -1
    top_b = top_b_list[0]["expert_id"] if top_b_list else -1
    return {
        "prompt_hash": prompt_hash,
        "epoch_a": int(epoch_a),
        "epoch_b": int(epoch_b),
        "token_position": int(token_position),
        "layer_idx": int(layer_idx),
        "route_rows_a": int(route_rows_a),
        "route_rows_b": int(route_rows_b),
        "selected_expert_count_a": int(len(selected_a)),
        "selected_expert_count_b": int(len(selected_b)),
        "cosine_similarity": _cosine_similarity(counts_a, counts_b),
        "jaccard_selected_experts": _jaccard(selected_a, selected_b),
        "shared_selected_expert_count": int(len(selected_a & selected_b)),
        "prev_only_expert_count": int(len(selected_a - selected_b)),
        "curr_only_expert_count": int(len(selected_b - selected_a)),
        "top1_match": int(top_a >= 0 and top_a == top_b),
    }


def _export_all_token_layer_similarity(*,
                                       artifact_root: Path,
                                       epoch_a: int,
                                       epoch_b: int,
                                       prompt_hash: str | None,
                                       prompt_index: int | None,
                                       output_csv: Path) -> Path:
    epoch_a_dir = _resolve_epoch_dir(artifact_root, epoch_a)
    epoch_b_dir = _resolve_epoch_dir(artifact_root, epoch_b)
    manifest_a = _load_manifest(epoch_a_dir)
    entry_a = _find_prompt_entry(manifest_a,
                                 prompt_hash=prompt_hash,
                                 prompt_index=prompt_index)
    payload_a = _load_prompt_payload(epoch_a_dir, entry_a)
    resolved_prompt_hash = entry_a["prompt_hash"]

    manifest_b = _load_manifest(epoch_b_dir)
    entry_b = _find_prompt_entry(manifest_b,
                                 prompt_hash=resolved_prompt_hash,
                                 prompt_index=None)
    payload_b = _load_prompt_payload(epoch_b_dir, entry_b)

    counts_a = payload_a["counts"]
    counts_b = payload_b["counts"]
    route_rows_a = payload_a["route_rows"]
    route_rows_b = payload_b["route_rows"]

    max_token_position = max(int(counts_a.shape[0]), int(counts_b.shape[0]))
    max_num_layers = max(int(counts_a.shape[1]), int(counts_b.shape[1]))

    fieldnames = [
        "prompt_hash",
        "epoch_a",
        "epoch_b",
        "token_position",
        "layer_idx",
        "route_rows_a",
        "route_rows_b",
        "selected_expert_count_a",
        "selected_expert_count_b",
        "cosine_similarity",
        "jaccard_selected_experts",
        "shared_selected_expert_count",
        "prev_only_expert_count",
        "curr_only_expert_count",
        "top1_match",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for token_position in range(max_token_position):
            for layer_idx in range(max_num_layers):
                counts_slice_a, route_slice_a = _safe_slice(payload_a,
                                                            token_position,
                                                            layer_idx)
                counts_slice_b, route_slice_b = _safe_slice(payload_b,
                                                            token_position,
                                                            layer_idx)
                writer.writerow(
                    _build_similarity_row(
                        prompt_hash=resolved_prompt_hash,
                        epoch_a=epoch_a,
                        epoch_b=epoch_b,
                        token_position=token_position,
                        layer_idx=layer_idx,
                        counts_a=counts_slice_a,
                        counts_b=counts_slice_b,
                        route_rows_a=route_slice_a,
                        route_rows_b=route_slice_b,
                    ))
                row_count += 1

    print(
        f"Wrote {row_count} rows for prompt_hash={resolved_prompt_hash} to {output_csv}"
    )
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze compact prompt-token-layer MoE artifacts.")
    parser.add_argument(
        "--artifact-root",
        default="moe_stats/prompt_token_layer_artifacts",
        help="Root directory that contains epoch_XXXX artifact folders.",
    )
    parser.add_argument("--epoch-a",
                        type=int,
                        required=True,
                        help="Primary epoch to inspect.")
    parser.add_argument("--epoch-b",
                        type=int,
                        default=None,
                        help="Optional comparison epoch.")
    parser.add_argument("--prompt-hash",
                        type=str,
                        default=None,
                        help="Prompt hash to inspect.")
    parser.add_argument("--prompt-index",
                        type=int,
                        default=None,
                        help="Prompt index from manifest ordering.")
    parser.add_argument("--token-position",
                        type=int,
                        default=None,
                        help="Token position to inspect.")
    parser.add_argument("--layer-idx",
                        type=int,
                        default=None,
                        help="Layer index to inspect.")
    parser.add_argument("--top-n",
                        type=int,
                        default=10,
                        help="How many top experts to print.")
    parser.add_argument("--list-prompts",
                        action="store_true",
                        help="List prompts in epoch-a and exit.")
    parser.add_argument("--limit",
                        type=int,
                        default=None,
                        help="Optional limit when listing prompts.")
    parser.add_argument("--json",
                        action="store_true",
                        help="Emit machine-readable JSON.")
    parser.add_argument(
        "--export-all-token-layer-csv",
        action="store_true",
        help=(
            "Export every token_position x layer similarity row for one prompt "
            "between epoch-a and epoch-b."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV path for --export-all-token-layer-csv.",
    )
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    epoch_a_dir = _resolve_epoch_dir(artifact_root, args.epoch_a)
    manifest_a = _load_manifest(epoch_a_dir)

    if args.list_prompts:
        _list_prompts(manifest_a, limit=args.limit)
        return

    if args.export_all_token_layer_csv:
        if args.epoch_b is None:
            raise ValueError(
                "--epoch-b is required when --export-all-token-layer-csv is used."
            )
        if args.prompt_hash is None and args.prompt_index is None:
            raise ValueError(
                "--prompt-hash or --prompt-index is required when "
                "--export-all-token-layer-csv is used."
            )
        output_csv = Path(
            args.output_csv
            if args.output_csv is not None else
            f"moe_stats/prompt_similarity_prompt_{args.prompt_index if args.prompt_index is not None else _sanitize_filename_fragment(args.prompt_hash)}_epoch_{int(args.epoch_a):04d}_vs_{int(args.epoch_b):04d}.csv"
        )
        _export_all_token_layer_similarity(
            artifact_root=artifact_root,
            epoch_a=int(args.epoch_a),
            epoch_b=int(args.epoch_b),
            prompt_hash=args.prompt_hash,
            prompt_index=args.prompt_index,
            output_csv=output_csv,
        )
        return

    if args.token_position is None or args.layer_idx is None:
        raise ValueError(
            "token_position and layer_idx are required unless --list-prompts is used."
        )

    entry_a = _find_prompt_entry(manifest_a,
                                 prompt_hash=args.prompt_hash,
                                 prompt_index=args.prompt_index)
    payload_a = _load_prompt_payload(epoch_a_dir, entry_a)
    prompt_hash = entry_a["prompt_hash"]
    counts_a, route_rows_a = _safe_slice(payload_a, args.token_position,
                                         args.layer_idx)
    selected_a = _selected_set(counts_a)

    result: dict[str, Any] = {
        "prompt_hash": prompt_hash,
        "token_position": int(args.token_position),
        "layer_idx": int(args.layer_idx),
        "epoch_a": {
            "epoch": int(args.epoch_a),
            "route_rows": int(route_rows_a),
            "selected_expert_count": int(len(selected_a)),
            "top_experts": _top_experts_from_counts(counts_a, args.top_n),
        },
    }

    if args.epoch_b is not None:
        epoch_b_dir = _resolve_epoch_dir(artifact_root, args.epoch_b)
        manifest_b = _load_manifest(epoch_b_dir)
        entry_b = _find_prompt_entry(manifest_b,
                                     prompt_hash=prompt_hash,
                                     prompt_index=None)
        payload_b = _load_prompt_payload(epoch_b_dir, entry_b)
        counts_b, route_rows_b = _safe_slice(payload_b, args.token_position,
                                             args.layer_idx)
        selected_b = _selected_set(counts_b)
        top_a = result["epoch_a"]["top_experts"][0][
            "expert_id"] if result["epoch_a"]["top_experts"] else -1
        top_b_list = _top_experts_from_counts(counts_b, args.top_n)
        top_b = top_b_list[0]["expert_id"] if top_b_list else -1
        result["epoch_b"] = {
            "epoch": int(args.epoch_b),
            "route_rows": int(route_rows_b),
            "selected_expert_count": int(len(selected_b)),
            "top_experts": top_b_list,
        }
        result["similarity"] = {
            "cosine_similarity": _cosine_similarity(counts_a, counts_b),
            "jaccard_selected_experts": _jaccard(selected_a, selected_b),
            "shared_selected_expert_count": int(len(selected_a & selected_b)),
            "prev_only_expert_count": int(len(selected_a - selected_b)),
            "curr_only_expert_count": int(len(selected_b - selected_a)),
            "top1_match": int(top_a >= 0 and top_a == top_b),
        }

    _emit(result, as_json=args.json)


if __name__ == "__main__":
    main()
