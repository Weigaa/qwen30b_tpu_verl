#!/usr/bin/env bash
set -euo pipefail

# Offline low-priority sidecar inference for resources released by elastic shrink.
# Required:
#   VERL_SIDECAR_MODEL_PATH=/path/to/model
# Optional:
#   VERL_SIDECAR_NPU_DEVICES=comma-separated released NPU ids
#   VERL_SIDECAR_PROMPTS_FILE=/path/to/prompts.txt|jsonl|json|parquet|dataset_dir
#   VERL_SIDECAR_OUTPUT_FILE=/path/to/output.jsonl
#   VERL_SIDECAR_MAX_SECONDS=60

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TS=$(date +%Y%m%d%H%M%S)

: "${VERL_SIDECAR_MODEL_PATH:?VERL_SIDECAR_MODEL_PATH must point to the sidecar inference model}"
: "${VERL_SIDECAR_NPU_DEVICES:?VERL_SIDECAR_NPU_DEVICES must be set by the shrink watcher or manually for direct use}"
VERL_SIDECAR_MASTER_PORT=${VERL_SIDECAR_MASTER_PORT:-24300}
VERL_SIDECAR_HCCL_IF_BASE_PORT=${VERL_SIDECAR_HCCL_IF_BASE_PORT:-52000}
VERL_SIDECAR_LOG_FILE=${VERL_SIDECAR_LOG_FILE:-"${ROOT_DIR}/sidecar_infer_${TS}.log"}
VERL_SIDECAR_OUTPUT_FILE=${VERL_SIDECAR_OUTPUT_FILE:-"${ROOT_DIR}/sidecar_infer_${TS}.jsonl"}
VERL_SIDECAR_MAX_SECONDS=${VERL_SIDECAR_MAX_SECONDS:-0}
VERL_SIDECAR_DEVICE_COUNT=$(python3 - "${VERL_SIDECAR_NPU_DEVICES}" <<'PY'
import sys

devices = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
print(len(devices))
PY
)
if [[ "${VERL_SIDECAR_DEVICE_COUNT}" == "0" ]]; then
    echo "VERL_SIDECAR_NPU_DEVICES does not contain any usable device id: ${VERL_SIDECAR_NPU_DEVICES}" >&2
    exit 2
fi
VERL_SIDECAR_PARALLEL_MODE=${VERL_SIDECAR_PARALLEL_MODE:-dp}
case "${VERL_SIDECAR_PARALLEL_MODE}" in
    dp)
        VERL_SIDECAR_TENSOR_PARALLEL_SIZE=${VERL_SIDECAR_TENSOR_PARALLEL_SIZE:-1}
        ;;
    tp)
        VERL_SIDECAR_TENSOR_PARALLEL_SIZE=${VERL_SIDECAR_TENSOR_PARALLEL_SIZE:-"${VERL_SIDECAR_DEVICE_COUNT}"}
        ;;
    *)
        echo "Unsupported VERL_SIDECAR_PARALLEL_MODE=${VERL_SIDECAR_PARALLEL_MODE}; expected dp or tp" >&2
        exit 2
        ;;
esac
VERL_SIDECAR_GPU_MEMORY_UTILIZATION=${VERL_SIDECAR_GPU_MEMORY_UTILIZATION:-0.90}
VERL_SIDECAR_MAX_MODEL_LEN=${VERL_SIDECAR_MAX_MODEL_LEN:-2048}
VERL_SIDECAR_MAX_NUM_SEQS=${VERL_SIDECAR_MAX_NUM_SEQS:-128}
VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS=${VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS:-65536}
VERL_SIDECAR_MAX_TOKENS=${VERL_SIDECAR_MAX_TOKENS:-1024}
VERL_SIDECAR_TEMPERATURE=${VERL_SIDECAR_TEMPERATURE:-0.0}
VERL_SIDECAR_TOP_P=${VERL_SIDECAR_TOP_P:-1.0}
VERL_SIDECAR_N=${VERL_SIDECAR_N:-1}
VERL_SIDECAR_TRUST_REMOTE_CODE=${VERL_SIDECAR_TRUST_REMOTE_CODE:-1}
VERL_SIDECAR_PROMPT=${VERL_SIDECAR_PROMPT:-"Explain elastic resource sharing in one sentence."}
VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE=${VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE:-"${VERL_SIDECAR_MAX_NUM_SEQS}"}
VERL_SIDECAR_MAX_PROMPTS=${VERL_SIDECAR_MAX_PROMPTS:-$((VERL_SIDECAR_DEVICE_COUNT * VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE))}
VERL_SIDECAR_REPEAT_UNTIL_KILLED=${VERL_SIDECAR_REPEAT_UNTIL_KILLED:-1}
VERL_SIDECAR_MAX_ITERATIONS=${VERL_SIDECAR_MAX_ITERATIONS:-0}
VERL_SIDECAR_ITERATION_SLEEP_SECONDS=${VERL_SIDECAR_ITERATION_SLEEP_SECONDS:-0}
VERL_SIDECAR_GENERATE_CHUNK_SIZE=${VERL_SIDECAR_GENERATE_CHUNK_SIZE:-32}
VERL_SIDECAR_STREAM_CHECKPOINT=${VERL_SIDECAR_STREAM_CHECKPOINT:-1}
VERL_SIDECAR_PARTIAL_SYNC_EVERY_STEPS=${VERL_SIDECAR_PARTIAL_SYNC_EVERY_STEPS:-0}
if [[ -z "${VERL_SIDECAR_STATE_DIR:-}" ]]; then
    VERL_SIDECAR_STATE_DIR=$(python3 - "${ROOT_DIR}" "${VERL_SIDECAR_MODEL_PATH}" "${VERL_SIDECAR_PROMPTS_FILE:-default}" "${VERL_SIDECAR_DATA_SPLIT:-train}" <<'PY'
import os
import re
import sys
from pathlib import Path

root, model, prompts, split = sys.argv[1:5]

def safe(value: str) -> str:
    name = Path(value).name or value
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return name or "default"

print(os.path.join(root, "sidecar_runs", "state",
                   f"{safe(model)}_{safe(prompts)}_{safe(split)}"))
PY
)
fi

mkdir -p "$(dirname "${VERL_SIDECAR_LOG_FILE}")" \
    "$(dirname "${VERL_SIDECAR_OUTPUT_FILE}")" \
    "${VERL_SIDECAR_STATE_DIR}"

export ASCEND_RT_VISIBLE_DEVICES="${VERL_SIDECAR_NPU_DEVICES}"
export MASTER_PORT="${VERL_SIDECAR_MASTER_PORT}"
export HCCL_IF_BASE_PORT="${VERL_SIDECAR_HCCL_IF_BASE_PORT}"
export VLLM_DP_MASTER_PORT="${VERL_SIDECAR_MASTER_PORT}"
export VERL_SIDECAR_LOG_FILE
export VERL_SIDECAR_OUTPUT_FILE
export VERL_SIDECAR_MODEL_PATH
export VERL_SIDECAR_PROMPTS_FILE
export VERL_SIDECAR_MAX_PROMPTS
export VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE
export VERL_SIDECAR_DEVICE_COUNT
export VERL_SIDECAR_PARALLEL_MODE
export VERL_SIDECAR_TENSOR_PARALLEL_SIZE
export VERL_SIDECAR_GPU_MEMORY_UTILIZATION
export VERL_SIDECAR_MAX_MODEL_LEN
export VERL_SIDECAR_MAX_NUM_SEQS
export VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS
export VERL_SIDECAR_MAX_TOKENS
export VERL_SIDECAR_TEMPERATURE
export VERL_SIDECAR_TOP_P
export VERL_SIDECAR_N
export VERL_SIDECAR_TRUST_REMOTE_CODE
export VERL_SIDECAR_PROMPT
export VERL_SIDECAR_REPEAT_UNTIL_KILLED
export VERL_SIDECAR_MAX_ITERATIONS
export VERL_SIDECAR_ITERATION_SLEEP_SECONDS
export VERL_SIDECAR_GENERATE_CHUNK_SIZE
export VERL_SIDECAR_STREAM_CHECKPOINT
export VERL_SIDECAR_PARTIAL_SYNC_EVERY_STEPS
export VERL_SIDECAR_STATE_DIR
export VERL_SIDECAR_DATA_SPLIT="${VERL_SIDECAR_DATA_SPLIT:-train}"
export VERL_SIDECAR_USE_SHORT_DATA="${VERL_SIDECAR_USE_SHORT_DATA:-0}"
export VERL_SIDECAR_ENFORCE_EAGER="${VERL_SIDECAR_ENFORCE_EAGER:-1}"
# Do not inherit the training rollout DP=16 into the sidecar unless explicitly requested.
export VLLM_DP_SIZE="${VERL_SIDECAR_DP_SIZE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_LOGGING_LEVEL="${VERL_SIDECAR_VLLM_LOGGING_LEVEL:-INFO}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export VLLM_ENABLE_EXPERT_PARALLEL="${VERL_SIDECAR_ENABLE_EXPERT_PARALLEL:-0}"

SIDECAR_SHARD_OUTPUTS=()
PY_SCRIPT=$(mktemp /tmp/elastic_sidecar_infer.XXXXXX.py)
cleanup_sidecar() {
    set +e
    if [[ "${#SIDECAR_SHARD_OUTPUTS[@]}" -gt 0 ]]; then
        : > "${VERL_SIDECAR_OUTPUT_FILE}"
        for shard_output in "${SIDECAR_SHARD_OUTPUTS[@]}"; do
            if [[ -f "${shard_output}" ]]; then
                cat "${shard_output}" >> "${VERL_SIDECAR_OUTPUT_FILE}"
            fi
        done
    fi
    rm -f "${PY_SCRIPT}"
}
trap cleanup_sidecar EXIT

cat > "${PY_SCRIPT}" <<'PY'
from copy import copy
import json
import os
import signal
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _prompt_to_text(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("prompt", "text", "content", "question"):
            if key in value:
                return _prompt_to_text(value[key])
        if "messages" in value:
            return _prompt_to_text(value["messages"])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, dict) and "content" in item:
                role = item.get("role")
                content = _prompt_to_text(item.get("content"))
                parts.append(f"{role}: {content}" if role else content)
            else:
                parts.append(_prompt_to_text(item))
        return "\n".join(part for part in parts if part)
    return str(value)


def _resolve_prompt_path(path: Path) -> Path:
    if not path.is_dir():
        return path
    split = os.environ.get("VERL_SIDECAR_DATA_SPLIT", "train").strip() or "train"
    use_short = _as_bool(os.environ.get("VERL_SIDECAR_USE_SHORT_DATA", "0"))
    splits = _dedupe([split, "train", "test"])
    candidates = []
    for item in splits:
        if use_short:
            candidates.extend([f"{item}_short.parquet", f"{item}.parquet"])
        else:
            candidates.extend([f"{item}.parquet", f"{item}_short.parquet"])
    for name in candidates:
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No supported prompt parquet found under dataset dir: {path}; "
        f"checked={candidates}")


def _record(prompt_id: int, prompt: str, source: str) -> dict:
    return {"prompt_id": int(prompt_id), "prompt": prompt, "source": source}


def _load_parquet_records(path: Path) -> list[dict]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("Reading parquet prompts requires pandas/pyarrow in the sidecar environment.") from exc
    df = pd.read_parquet(path)
    records = []
    for row_idx, row in df.iterrows():
        prompt = ""
        if "prompt" in row and row["prompt"] is not None:
            prompt = _prompt_to_text(row["prompt"])
        elif "text" in row and row["text"] is not None:
            prompt = _prompt_to_text(row["text"])
        elif "question" in row and row["question"] is not None:
            prompt = _prompt_to_text(row["question"])
        else:
            extra_info = row.get("extra_info") if "extra_info" in row else None
            if isinstance(extra_info, dict) and extra_info.get("question") is not None:
                prompt = _prompt_to_text(extra_info["question"])
        if prompt:
            records.append(_record(len(records), prompt, f"{path}:{row_idx}"))
    return records


def load_prompt_records() -> list[dict]:
    prompt_file = os.environ.get("VERL_SIDECAR_PROMPTS_FILE", "").strip()
    fallback = os.environ.get("VERL_SIDECAR_PROMPT", "Explain elastic resource sharing in one sentence.")
    if not prompt_file:
        return [_record(0, fallback, "fallback")]
    path = _resolve_prompt_path(Path(prompt_file))
    if not path.exists():
        raise FileNotFoundError(f"VERL_SIDECAR_PROMPTS_FILE does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return _load_parquet_records(path)
    text = path.read_text(encoding="utf-8")
    records = []
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    prompt = item
                elif isinstance(item, dict):
                    prompt = _prompt_to_text(item.get("prompt", item.get("text", item)))
                else:
                    prompt = str(item)
                if prompt:
                    records.append(_record(len(records), prompt, f"{path}:{len(records)}"))
            return records
        if isinstance(data, dict):
            value = data.get("prompts", data.get("prompt", data.get("text", fallback)))
            if isinstance(value, list):
                for item in value:
                    prompt = _prompt_to_text(item)
                    if prompt:
                        records.append(_record(len(records), prompt, f"{path}:{len(records)}"))
                return records
            return [_record(0, _prompt_to_text(value), str(path))]
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("{"):
            obj = json.loads(line)
            prompt = _prompt_to_text(obj.get("prompt", obj.get("text", obj)))
        else:
            prompt = line
        if prompt:
            records.append(_record(len(records), prompt, f"{path}:{len(records)}"))
    return records


def _read_jsonl_ids(path: Path) -> set[int]:
    ids = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("prompt_id") is not None:
                ids.add(int(item["prompt_id"]))
    return ids


def _read_inflight_ids(path: Path) -> list[int]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    ids = data.get("prompt_ids", [])
    return [int(item) for item in ids if item is not None]


def _read_resume_records(path: Path) -> dict[int, dict]:
    records = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("prompt_id") is not None:
                records[int(item["prompt_id"])] = item
    return records


def _write_resume_records(path: Path, records: dict[int, dict]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        for prompt_id in sorted(records):
            f.write(json.dumps(records[prompt_id], ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _stable_records(records: list[dict]) -> list[dict]:
    offset = int(os.environ.get("VERL_SIDECAR_PROMPT_OFFSET", "0"))
    if records and offset > 0:
        offset = offset % len(records)
        records = records[offset:] + records[:offset]
    return records


def _select_pending_records(records: list[dict], completed_ids: set[int],
                            inflight_ids: list[int], resume_records: dict[int, dict],
                            max_records: int) -> list[dict]:
    by_id = {int(item["prompt_id"]): item for item in records}
    shard_records = [
        item for item in _stable_records(records)
        if int(item["prompt_id"]) % num_shards == shard_index
    ]
    selected = []
    seen = set()
    priority_ids = list(inflight_ids) + sorted(resume_records)
    for prompt_id in priority_ids:
        if prompt_id in completed_ids or prompt_id in seen:
            continue
        item = by_id.get(prompt_id)
        if item is not None and int(item["prompt_id"]) % num_shards == shard_index:
            resume = resume_records.get(prompt_id)
            item = dict(item)
            if resume:
                item["resume_prompt"] = resume.get("resume_prompt", item["prompt"])
                item["resume_prefix_text"] = resume.get("partial_text", "")
                item["resume_token_ids_len"] = int(resume.get("token_ids_len", 0) or 0)
            selected.append(item)
            seen.add(prompt_id)
    for item in shard_records:
        prompt_id = int(item["prompt_id"])
        if prompt_id in completed_ids or prompt_id in seen:
            continue
        selected.append(item)
        seen.add(prompt_id)
        if max_records > 0 and len(selected) >= max_records:
            break
    if max_records > 0:
        selected = selected[:max_records]
    return selected


def _chunks(items: list[dict], size: int):
    if size <= 0:
        size = len(items) or 1
    for start in range(0, len(items), size):
        yield start, items[start:start + size]


_shutdown_requested = False


def _request_shutdown(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _request_shutdown)
signal.signal(signal.SIGINT, _request_shutdown)


def _completion_payload(out, prefix_text: str = "") -> tuple[dict, int, str, int]:
    token_ids = getattr(out, "token_ids", None) or []
    text = prefix_text + out.text
    payload = {
        "text": text,
        "delta_text": out.text,
        "token_ids_len": len(token_ids),
        "resume_prefix_text_len": len(prefix_text),
        "finish_reason": getattr(out, "finish_reason", None),
    }
    return payload, len(token_ids), text, len(token_ids)


def _write_final_output(output_file: Path, completed_file: Path, resume_file: Path,
                        resume_records: dict[int, dict], source: dict,
                        output, completions: list[dict], iteration: int,
                        chunk_start: int) -> None:
    prompt_id = int(source["prompt_id"])
    with output_file.open("a", encoding="utf-8") as out_f:
        out_f.write(json.dumps({
            "iteration": iteration,
            "chunk_start": chunk_start,
            "prompt_id": prompt_id,
            "prompt_source": source.get("source", ""),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "prompt": source["prompt"],
            "resume_prompt": output.prompt,
            "prompt_token_ids_len": len(getattr(output, "prompt_token_ids", None) or []),
            "outputs": completions,
        }, ensure_ascii=False) + "\n")
        out_f.flush()
        os.fsync(out_f.fileno())
    _append_jsonl(completed_file, [{
        "time": time.time(),
        "iteration": iteration,
        "chunk_start": chunk_start,
        "prompt_id": prompt_id,
        "output_file": str(output_file),
    }])
    resume_records.pop(prompt_id, None)
    _write_resume_records(resume_file, resume_records)


def _generate_chunk_blocking(llm, chunk_records: list[dict], sampling_params,
                             output_file: Path, completed_file: Path,
                             resume_file: Path, resume_records: dict[int, dict],
                             iteration: int, chunk_start: int) -> tuple[float, int, int]:
    infer_start = time.perf_counter()
    outputs = llm.generate([item.get("resume_prompt", item["prompt"]) for item in chunk_records],
                           sampling_params, use_tqdm=False)
    infer_s = time.perf_counter() - infer_start
    chunk_output_tokens = 0
    for output_idx, output in enumerate(outputs):
        source = chunk_records[output_idx] if output_idx < len(chunk_records) else None
        if source is None:
            continue
        prefix_text = source.get("resume_prefix_text", "")
        completions = []
        for out in output.outputs:
            payload, token_count, _, _ = _completion_payload(out, prefix_text)
            chunk_output_tokens += token_count
            completions.append(payload)
        _write_final_output(output_file, completed_file, resume_file, resume_records,
                            source, output, completions, iteration, chunk_start)
    return infer_s, chunk_output_tokens, len(outputs)


def _generate_chunk_streaming(llm, chunk_records: list[dict], sampling_params,
                              output_file: Path, completed_file: Path,
                              resume_file: Path, partials_file: Path,
                              resume_records: dict[int, dict],
                              iteration: int, chunk_start: int) -> tuple[float, int, int]:
    request_to_record = {}
    partial_rows = []
    latest_by_request = {}
    for item in chunk_records:
        params = copy(sampling_params)
        params.output_kind = RequestOutputKind.CUMULATIVE
        request_id = str(next(llm.request_counter))
        prompt = item.get("resume_prompt", item["prompt"])
        llm.llm_engine.add_request(request_id, prompt, params, tokenization_kwargs={})
        request_to_record[request_id] = item

    infer_start = time.perf_counter()
    chunk_output_tokens = 0
    finished_requests = 0
    step_idx = 0
    partial_sync_every_steps = int(os.environ.get("VERL_SIDECAR_PARTIAL_SYNC_EVERY_STEPS", "0"))

    while request_to_record:
        step_outputs = llm.llm_engine.step()
        step_idx += 1
        for output in step_outputs:
            request_id = str(output.request_id)
            source = request_to_record.get(request_id)
            if source is None:
                continue
            prompt_id = int(source["prompt_id"])
            prefix_text = source.get("resume_prefix_text", "")
            completions = []
            latest_text = prefix_text
            latest_tokens = int(source.get("resume_token_ids_len", 0) or 0)
            step_tokens = 0
            for out in output.outputs:
                payload, token_count, full_text, latest_token_count = _completion_payload(out, prefix_text)
                completions.append(payload)
                step_tokens += token_count
                latest_text = full_text
                latest_tokens = latest_token_count
            partial_row = {
                "time": time.time(),
                "iteration": iteration,
                "chunk_start": chunk_start,
                "request_id": request_id,
                "prompt_id": prompt_id,
                "shard_index": shard_index,
                "num_shards": num_shards,
                "finished": bool(output.finished),
                "prompt": source["prompt"],
                "resume_prompt": output.prompt,
                "outputs": completions,
            }
            partial_rows.append(partial_row)
            latest_by_request[request_id] = partial_row

            if output.finished:
                _write_final_output(output_file, completed_file, resume_file,
                                    resume_records, source, output, completions,
                                    iteration, chunk_start)
                request_to_record.pop(request_id, None)
                latest_by_request.pop(request_id, None)
                finished_requests += 1
                chunk_output_tokens += step_tokens
            else:
                resume_records[prompt_id] = {
                    "time": time.time(),
                    "iteration": iteration,
                    "chunk_start": chunk_start,
                    "prompt_id": prompt_id,
                    "prompt": source["prompt"],
                    "resume_prompt": source["prompt"] + latest_text,
                    "partial_text": latest_text,
                    "token_ids_len": latest_tokens,
                    "request_id": request_id,
                }

        if partial_sync_every_steps > 0 and step_idx % partial_sync_every_steps == 0:
            _append_jsonl(partials_file, partial_rows)
            _write_resume_records(resume_file, resume_records)
            partial_rows.clear()

        if _shutdown_requested:
            if latest_by_request:
                _append_jsonl(partials_file, list(latest_by_request.values()))
            _write_resume_records(resume_file, resume_records)
            if request_to_record:
                try:
                    llm.llm_engine.abort_request(list(request_to_record))
                except Exception:
                    pass
            break

    if partial_rows and (partial_sync_every_steps > 0 or _shutdown_requested):
        _append_jsonl(partials_file, partial_rows)
    if _shutdown_requested:
        _write_resume_records(resume_file, resume_records)

    infer_s = time.perf_counter() - infer_start
    return infer_s, chunk_output_tokens, finished_requests

model_path = os.environ["VERL_SIDECAR_MODEL_PATH"]
output_file = Path(os.environ["VERL_SIDECAR_OUTPUT_FILE"])
output_file.parent.mkdir(parents=True, exist_ok=True)

shard_index = int(os.environ.get("VERL_SIDECAR_SHARD_INDEX", "0"))
num_shards = int(os.environ.get("VERL_SIDECAR_NUM_SHARDS", "1"))
if shard_index < 0 or num_shards <= 0 or shard_index >= num_shards:
    raise ValueError(f"Invalid sidecar shard: index={shard_index}, num_shards={num_shards}")
repeat_until_killed = _as_bool(os.environ.get("VERL_SIDECAR_REPEAT_UNTIL_KILLED", "1"))
max_iterations = int(os.environ.get("VERL_SIDECAR_MAX_ITERATIONS", "0"))
iteration_sleep_s = float(os.environ.get("VERL_SIDECAR_ITERATION_SLEEP_SECONDS", "0"))
max_prompts = int(os.environ.get("VERL_SIDECAR_MAX_PROMPTS", "32"))
max_prompts_per_device = int(os.environ.get("VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE", "0"))
if num_shards > 1:
    max_records_per_iteration = max_prompts_per_device or ((max_prompts + num_shards - 1) // num_shards)
else:
    max_records_per_iteration = max_prompts
chunk_size = int(os.environ.get("VERL_SIDECAR_GENERATE_CHUNK_SIZE", "32"))
stream_checkpoint = _as_bool(os.environ.get("VERL_SIDECAR_STREAM_CHECKPOINT", "1"))

state_dir = Path(os.environ.get("VERL_SIDECAR_STATE_DIR", str(output_file.parent / "state")))
state_dir.mkdir(parents=True, exist_ok=True)
completed_file = state_dir / f"completed.shard{shard_index}.jsonl"
inflight_file = state_dir / f"inflight.shard{shard_index}.json"
resume_file = state_dir / f"resume.shard{shard_index}.jsonl"
partials_file = state_dir / f"partials.shard{shard_index}.jsonl"

records = load_prompt_records()
completed_ids = _read_jsonl_ids(completed_file)
inflight_ids = _read_inflight_ids(inflight_file)
resume_records = _read_resume_records(resume_file)
initial_records = _select_pending_records(records, completed_ids, inflight_ids,
                                          resume_records, max_records_per_iteration)
if not initial_records:
    output_file.touch()
    print(json.dumps({
        "event": "sidecar_no_work",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "total_prompts": len(records),
        "completed_prompts": len(completed_ids),
        "resume_prompts": len(resume_records),
        "state_dir": str(state_dir),
        "output_file": str(output_file),
    }, ensure_ascii=False), flush=True)
    raise SystemExit(0)
if output_file.exists() and _as_bool(os.environ.get("VERL_SIDECAR_RESET_OUTPUT_ON_START", "0")):
    output_file.unlink()

start_total = time.perf_counter()
load_start = time.perf_counter()
engine_kwargs = {
    "model": model_path,
    "tensor_parallel_size": int(os.environ.get("VERL_SIDECAR_TENSOR_PARALLEL_SIZE", "1")),
    "gpu_memory_utilization": float(os.environ.get("VERL_SIDECAR_GPU_MEMORY_UTILIZATION", "0.80")),
    "max_num_seqs": int(os.environ.get("VERL_SIDECAR_MAX_NUM_SEQS", "16")),
    "max_num_batched_tokens": int(os.environ.get("VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS", "1024")),
    "trust_remote_code": _as_bool(os.environ.get("VERL_SIDECAR_TRUST_REMOTE_CODE", "1")),
    "enforce_eager": _as_bool(os.environ.get("VERL_SIDECAR_ENFORCE_EAGER", "1")),
}
max_model_len = os.environ.get("VERL_SIDECAR_MAX_MODEL_LEN", "").strip()
if max_model_len:
    engine_kwargs["max_model_len"] = int(max_model_len)

print(json.dumps({
    "event": "sidecar_load_start",
    "model_path": model_path,
    "devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
    "master_port": os.environ.get("MASTER_PORT", ""),
    "hccl_if_base_port": os.environ.get("HCCL_IF_BASE_PORT", ""),
    "num_prompts": len(initial_records),
    "total_prompts": len(records),
    "completed_prompts": len(completed_ids),
    "inflight_prompts": len(inflight_ids),
    "resume_prompts": len(resume_records),
    "stream_checkpoint": stream_checkpoint,
    "shard_index": shard_index,
    "num_shards": num_shards,
    "repeat_until_killed": repeat_until_killed,
    "max_iterations": max_iterations,
    "prompts_source": os.environ.get("VERL_SIDECAR_PROMPTS_FILE", ""),
    "data_split": os.environ.get("VERL_SIDECAR_DATA_SPLIT", ""),
    "use_short_data": os.environ.get("VERL_SIDECAR_USE_SHORT_DATA", ""),
    "state_dir": str(state_dir),
    "resume_file": str(resume_file),
    "partials_file": str(partials_file),
    "max_prompts": max_prompts,
    "max_prompts_per_device": max_prompts_per_device,
    "max_records_per_iteration": max_records_per_iteration,
    "generate_chunk_size": chunk_size,
    "engine_kwargs": {k: v for k, v in engine_kwargs.items() if k != "model"},
}, ensure_ascii=False), flush=True)

llm = LLM(**engine_kwargs)
load_s = time.perf_counter() - load_start

sampling_params = SamplingParams(
    n=int(os.environ.get("VERL_SIDECAR_N", "1")),
    temperature=float(os.environ.get("VERL_SIDECAR_TEMPERATURE", "0.0")),
    top_p=float(os.environ.get("VERL_SIDECAR_TOP_P", "1.0")),
    max_tokens=int(os.environ.get("VERL_SIDECAR_MAX_TOKENS", "1024")),
)

iteration = 0
total_requests = 0
total_output_tokens = 0
total_infer_s = 0.0
last_total_prompts = len(records)
while True:
    completed_ids = _read_jsonl_ids(completed_file)
    inflight_ids = _read_inflight_ids(inflight_file)
    resume_records = _read_resume_records(resume_file)
    records = load_prompt_records()
    last_total_prompts = len(records)
    selected_records = _select_pending_records(records, completed_ids, inflight_ids,
                                               resume_records, max_records_per_iteration)
    prompt_offset = int(os.environ.get("VERL_SIDECAR_PROMPT_OFFSET", "0"))
    if not selected_records:
        print(json.dumps({
            "event": "sidecar_iteration_no_work",
            "iteration": iteration,
            "prompt_offset": prompt_offset,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "total_prompts": len(records),
            "completed_prompts": len(completed_ids),
            "resume_prompts": len(resume_records),
            "state_dir": str(state_dir),
            "output_file": str(output_file),
        }, ensure_ascii=False), flush=True)
        break

    iteration_requests = 0
    iteration_output_tokens = 0
    iteration_infer_s = 0.0
    for chunk_start, chunk_records in _chunks(selected_records, chunk_size):
        chunk_ids = [int(item["prompt_id"]) for item in chunk_records]
        _atomic_write_json(inflight_file, {
            "time": time.time(),
            "iteration": iteration,
            "chunk_start": chunk_start,
            "prompt_ids": chunk_ids,
            "shard_index": shard_index,
            "num_shards": num_shards,
        })
        print(json.dumps({
            "event": "sidecar_chunk_start",
            "iteration": iteration,
            "chunk_start": chunk_start,
            "chunk_size": len(chunk_records),
            "prompt_ids_first": chunk_ids[:8],
            "stream_checkpoint": stream_checkpoint,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "state_dir": str(state_dir),
        }, ensure_ascii=False), flush=True)

        resume_records = _read_resume_records(resume_file)
        if stream_checkpoint:
            infer_s, chunk_output_tokens, finished_requests = _generate_chunk_streaming(
                llm, chunk_records, sampling_params, output_file, completed_file,
                resume_file, partials_file, resume_records, iteration, chunk_start)
        else:
            infer_s, chunk_output_tokens, finished_requests = _generate_chunk_blocking(
                llm, chunk_records, sampling_params, output_file, completed_file,
                resume_file, resume_records, iteration, chunk_start)
        try:
            inflight_file.unlink()
        except FileNotFoundError:
            pass

        iteration_requests += finished_requests
        iteration_output_tokens += chunk_output_tokens
        iteration_infer_s += infer_s
        print(json.dumps({
            "event": "sidecar_chunk_done",
            "iteration": iteration,
            "chunk_start": chunk_start,
            "inference_time_s": infer_s,
            "num_requests": finished_requests,
            "submitted_requests": len(chunk_records),
            "prompt_ids_first": chunk_ids[:8],
            "shutdown_requested": _shutdown_requested,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "num_output_tokens": chunk_output_tokens,
            "tokens_per_s": (chunk_output_tokens / infer_s) if infer_s > 0 else 0.0,
            "completed_file": str(completed_file),
            "resume_file": str(resume_file),
            "output_file": str(output_file),
        }, ensure_ascii=False), flush=True)
        if _shutdown_requested:
            break

    total_requests += iteration_requests
    total_output_tokens += iteration_output_tokens
    total_infer_s += iteration_infer_s
    print(json.dumps({
        "event": "sidecar_iteration_done",
        "iteration": iteration,
        "prompt_offset": prompt_offset,
        "inference_time_s": iteration_infer_s,
        "num_requests": iteration_requests,
        "total_prompts": len(records),
        "completed_prompts": len(_read_jsonl_ids(completed_file)),
        "resume_prompts": len(_read_resume_records(resume_file)),
        "shutdown_requested": _shutdown_requested,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "num_output_tokens": iteration_output_tokens,
        "tokens_per_s": (iteration_output_tokens / iteration_infer_s) if iteration_infer_s > 0 else 0.0,
        "state_dir": str(state_dir),
        "output_file": str(output_file),
    }, ensure_ascii=False), flush=True)

    iteration += 1
    if _shutdown_requested:
        break
    if not repeat_until_killed:
        break
    if max_iterations > 0 and iteration >= max_iterations:
        break
    if iteration_sleep_s > 0:
        time.sleep(iteration_sleep_s)

total_s = time.perf_counter() - start_total
print(json.dumps({
    "event": "sidecar_done",
    "model_load_time_s": load_s,
    "inference_time_s": total_infer_s,
    "total_time_s": total_s,
    "num_requests": total_requests,
    "iterations": iteration,
    "total_prompts_last_iteration": last_total_prompts,
    "completed_prompts": len(_read_jsonl_ids(completed_file)),
    "resume_prompts": len(_read_resume_records(resume_file)),
    "shutdown_requested": _shutdown_requested,
    "shard_index": shard_index,
    "num_shards": num_shards,
    "num_output_tokens": total_output_tokens,
    "tokens_per_s": (total_output_tokens / total_infer_s) if total_infer_s > 0 else 0.0,
    "state_dir": str(state_dir),
    "output_file": str(output_file),
}, ensure_ascii=False), flush=True)

PY

{
    echo "sidecar_start_time=$(date +%s.%N)"
    echo "sidecar_devices=${VERL_SIDECAR_NPU_DEVICES}"
    echo "sidecar_device_count=${VERL_SIDECAR_DEVICE_COUNT}"
    echo "sidecar_parallel_mode=${VERL_SIDECAR_PARALLEL_MODE}"
    echo "sidecar_tensor_parallel_size=${VERL_SIDECAR_TENSOR_PARALLEL_SIZE}"
    echo "sidecar_model=${VERL_SIDECAR_MODEL_PATH}"
    echo "sidecar_output=${VERL_SIDECAR_OUTPUT_FILE}"
    echo "sidecar_state_dir=${VERL_SIDECAR_STATE_DIR}"
    echo "sidecar_generate_chunk_size=${VERL_SIDECAR_GENERATE_CHUNK_SIZE}"
    echo "sidecar_stream_checkpoint=${VERL_SIDECAR_STREAM_CHECKPOINT}"
    echo "sidecar_partial_sync_every_steps=${VERL_SIDECAR_PARTIAL_SYNC_EVERY_STEPS}"
    set +e
    if [[ "${VERL_SIDECAR_PARALLEL_MODE}" == "dp" && "${VERL_SIDECAR_DEVICE_COUNT}" -gt 1 ]]; then
        IFS=',' read -r -a sidecar_devices <<< "${VERL_SIDECAR_NPU_DEVICES}"
        sidecar_pids=()
        shard_index=0
        for raw_device in "${sidecar_devices[@]}"; do
            device=$(echo "${raw_device}" | xargs)
            [[ -n "${device}" ]] || continue
            shard_output="${VERL_SIDECAR_OUTPUT_FILE}.shard${shard_index}"
            SIDECAR_SHARD_OUTPUTS+=("${shard_output}")
            (
                export ASCEND_RT_VISIBLE_DEVICES="${device}"
                export VERL_SIDECAR_SHARD_INDEX="${shard_index}"
                export VERL_SIDECAR_NUM_SHARDS="${VERL_SIDECAR_DEVICE_COUNT}"
                export VERL_SIDECAR_OUTPUT_FILE="${shard_output}"
                export MASTER_PORT=$((VERL_SIDECAR_MASTER_PORT + shard_index))
                export HCCL_IF_BASE_PORT=$((VERL_SIDECAR_HCCL_IF_BASE_PORT + shard_index * 16))
                export VLLM_DP_MASTER_PORT="${MASTER_PORT}"
                echo "sidecar_shard_start_time=$(date +%s.%N) shard=${shard_index} device=${device} output=${shard_output}"
                if [[ "${VERL_SIDECAR_MAX_SECONDS}" != "0" ]]; then
                    timeout --kill-after=10s "${VERL_SIDECAR_MAX_SECONDS}s" python3 -u "${PY_SCRIPT}" 2>&1
                    shard_rc=$?
                else
                    python3 -u "${PY_SCRIPT}" 2>&1
                    shard_rc=$?
                fi
                echo "sidecar_shard_end_time=$(date +%s.%N) shard=${shard_index} device=${device} exit_code=${shard_rc}"
                exit "${shard_rc}"
            ) &
            sidecar_pids+=("$!")
            shard_index=$((shard_index + 1))
        done
        rc=0
        for pid in "${sidecar_pids[@]}"; do
            wait "${pid}"
            shard_wait_rc=$?
            if [[ "${shard_wait_rc}" != "0" && "${rc}" == "0" ]]; then
                rc="${shard_wait_rc}"
            fi
        done
        : > "${VERL_SIDECAR_OUTPUT_FILE}"
        for shard_output in "${SIDECAR_SHARD_OUTPUTS[@]}"; do
            if [[ -f "${shard_output}" ]]; then
                cat "${shard_output}" >> "${VERL_SIDECAR_OUTPUT_FILE}"
            fi
        done
    elif [[ "${VERL_SIDECAR_MAX_SECONDS}" != "0" ]]; then
        timeout --kill-after=10s "${VERL_SIDECAR_MAX_SECONDS}s" python3 -u "${PY_SCRIPT}" 2>&1
        rc=$?
    else
        python3 -u "${PY_SCRIPT}" 2>&1
        rc=$?
    fi
    set -e
    echo "sidecar_end_time=$(date +%s.%N)"
    echo "sidecar_exit_code=${rc}"
    if [[ "${rc}" == "124" || "${rc}" == "137" ]]; then
        echo "sidecar_killed_by_deadline=1"
    else
        echo "sidecar_killed_by_deadline=0"
    fi
    exit "${rc}"
} | tee -a "${VERL_SIDECAR_LOG_FILE}"
