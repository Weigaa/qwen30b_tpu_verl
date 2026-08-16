#!/usr/bin/env python3
"""Validate DeepSeek-V2-Lite HF assets and an optional MCore checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jinja2 import Environment, StrictUndefined, TemplateError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from verl.utils.deepseek_v2_lite import (  # noqa: E402
    ARCHITECTURE,
    MODEL_ID,
    MODEL_REVISION,
    local_routed_experts_by_floor,
    validate_config,
)


def validate_chat_template(tokenizer_config: object) -> list[str]:
    """Validate the role and termination contract of the pinned Chat tokenizer."""

    if not isinstance(tokenizer_config, dict):
        return ["tokenizer_config.json must contain a JSON object"]
    template = tokenizer_config.get("chat_template")
    if not isinstance(template, str) or not template.strip():
        return ["tokenizer_config.json has no nonempty chat_template"]
    required_fragments = (
        "add_generation_prompt",
        "message['role'] == 'user'",
        "message['role'] == 'assistant'",
        "Assistant:",
        "eos_token",
    )
    missing = [
        fragment for fragment in required_fragments if fragment not in template
    ]
    if missing:
        return [f"chat_template is missing required fragment {fragment!r}" for fragment in missing]
    bos = "<ADAFLOOR_BOS>"
    eos = "<ADAFLOOR_EOS>"
    user = "ADAFLOOR_USER_SENTINEL"
    assistant = "ADAFLOOR_ASSISTANT_SENTINEL"
    try:
        compiled = Environment(undefined=StrictUndefined).from_string(template)
        single = compiled.render(
            messages=[{"role": "user", "content": user}],
            bos_token=bos,
            eos_token=eos,
            add_generation_prompt=True,
        )
        history = compiled.render(
            messages=[
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
                {"role": "user", "content": "ADAFLOOR_SECOND_USER"},
            ],
            bos_token=bos,
            eos_token=eos,
            add_generation_prompt=True,
        )
    except TemplateError as exc:
        return [f"chat_template cannot be rendered: {exc}"]
    errors: list[str] = []
    if not single.startswith(bos) or single.count(bos) != 1:
        errors.append("chat_template must render exactly one leading BOS token")
    if single.count(user) != 1:
        errors.append("chat_template must render user content exactly once")
    if not single.rstrip().endswith("Assistant:"):
        errors.append("chat_template must end a generation prompt with Assistant:")
    if f"{assistant}{eos}" not in history:
        errors.append("chat_template must terminate assistant history with EOS")
    if not history.rstrip().endswith("Assistant:"):
        errors.append("chat_template history must end with a new assistant prompt")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--distcp-path", type=Path)
    parser.add_argument(
        "--allow-missing-weights",
        action="store_true",
        help="Validate configuration and tokenizer files before weight download completes.",
    )
    parser.add_argument("--expected-revision", default=MODEL_REVISION)
    parser.add_argument("--expected-model-id", default=MODEL_ID)
    parser.add_argument("--expected-pp-size", type=int, default=4)
    parser.add_argument("--expected-ep-size", type=int, default=4)
    return parser.parse_args()


def _download_revisions(model_path: Path) -> tuple[list[Path], set[str]]:
    metadata_root = model_path / ".cache" / "huggingface" / "download"
    metadata_files = sorted(metadata_root.rglob("*.metadata"))
    revisions: set[str] = set()
    for path in metadata_files:
        try:
            revision = path.read_text(encoding="utf-8").splitlines()[0].strip()
        except (OSError, IndexError):
            continue
        if revision:
            revisions.add(revision)
    return metadata_files, revisions


def validate_hf_assets(
    model_path: Path,
    allow_missing_weights: bool,
    expected_revision: str,
    expected_model_id: str,
) -> dict:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"missing {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid config.json: {exc}") from exc

    errors = validate_config(config)
    tokenizer_candidates = (
        model_path / "tokenizer.json",
        model_path / "tokenizer.model",
    )
    if not any(path.is_file() for path in tokenizer_candidates):
        errors.append("missing tokenizer.json or tokenizer.model")
    tokenizer_config_path = model_path / "tokenizer_config.json"
    if not tokenizer_config_path.is_file():
        errors.append("missing tokenizer_config.json")
    else:
        try:
            tokenizer_config = json.loads(
                tokenizer_config_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid tokenizer_config.json: {exc}")
        else:
            errors.extend(validate_chat_template(tokenizer_config))

    weight_index = model_path / "model.safetensors.index.json"
    weight_files: set[Path] = set()
    if weight_index.is_file():
        try:
            index = json.loads(weight_index.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("weight_map is empty")
            weight_files = {model_path / str(name) for name in weight_map.values()}
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            errors.append(f"invalid model.safetensors.index.json: {exc}")
    else:
        weight_files = set(model_path.glob("*.safetensors"))

    if not allow_missing_weights:
        if not weight_files:
            errors.append("no safetensors weights found")
        missing = sorted(path.name for path in weight_files if not path.is_file())
        if missing:
            errors.append(f"missing weight shards: {missing[:8]}")
        empty = sorted(
            path.name for path in weight_files if path.is_file() and path.stat().st_size == 0
        )
        if empty:
            errors.append(f"empty weight shards: {empty[:8]}")

    metadata_files, revisions = _download_revisions(model_path)
    if not metadata_files:
        errors.append("missing Hugging Face local download metadata")
    elif revisions != {expected_revision}:
        errors.append(
            "download revision mismatch: "
            f"expected {expected_revision}, observed {sorted(revisions)}"
        )

    if errors:
        raise ValueError("\n  - ".join(["asset validation failed", *errors]))
    return {
        "model_id": expected_model_id,
        "model_revision": expected_revision,
        "architecture": ARCHITECTURE,
        "model_path": str(model_path.resolve()),
        "weight_shards": len(weight_files),
        "download_metadata_files": len(metadata_files),
        "downloaded_revisions": sorted(revisions),
        "local_routed_experts": local_routed_experts_by_floor(config),
    }


def validate_distcp(
    distcp_path: Path,
    expected_model_id: str,
    expected_revision: str,
    expected_pp_size: int,
    expected_ep_size: int,
) -> dict:
    if not distcp_path.is_dir():
        raise ValueError(f"missing MCore checkpoint directory: {distcp_path}")
    shards = sorted(distcp_path.rglob("*.distcp"))
    metadata = sorted(distcp_path.rglob(".metadata"))
    if not shards:
        raise ValueError(f"no .distcp shards found under {distcp_path}")
    if any(path.stat().st_size == 0 for path in shards):
        raise ValueError(f"empty .distcp shard found under {distcp_path}")
    if not metadata:
        raise ValueError(f"no distributed-checkpoint metadata found under {distcp_path}")

    manifest_path = distcp_path / ".adafloor_deepseek_v2_lite_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing conversion manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid conversion manifest: {exc}") from exc
    expected = {
        "model_id": expected_model_id,
        "model_revision": expected_revision,
        "architecture": ARCHITECTURE,
        "pipeline_model_parallel_size": expected_pp_size,
        "expert_model_parallel_size": expected_ep_size,
        "world_size": expected_pp_size * expected_ep_size,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"conversion manifest mismatch: {mismatches}")
    return {
        "distcp_path": str(distcp_path.resolve()),
        "distcp_shards": len(shards),
        "metadata_files": len(metadata),
        "conversion_manifest": str(manifest_path),
        "pipeline_model_parallel_size": expected_pp_size,
        "expert_model_parallel_size": expected_ep_size,
    }


def main() -> int:
    args = parse_args()
    try:
        report = validate_hf_assets(
            args.model_path,
            args.allow_missing_weights,
            args.expected_revision,
            args.expected_model_id,
        )
        if args.distcp_path is not None:
            report.update(
                validate_distcp(
                    args.distcp_path,
                    args.expected_model_id,
                    args.expected_revision,
                    args.expected_pp_size,
                    args.expected_ep_size,
                )
            )
    except ValueError as exc:
        print(f"[DeepSeek-V2-Lite validate] {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
