#!/usr/bin/env python3
"""Verify a helper's *real* NPU allocation behavior outside timed workloads."""
import argparse
import importlib.util
import json

import torch
import torch_npu  # noqa: F401 -- registers the NPU backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("helper")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    spec = importlib.util.spec_from_file_location("allocator_helper", args.helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    records = []
    for enabled in (False, True, False):
        helper.set_expandable_segments(enabled)
        # A fresh stream prevents reuse of a segment allocated in a previous
        # mode. Do not infer runtime settings from the environment variable.
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            tensor = torch.empty(4 * 1024 * 1024, device="npu", dtype=torch.uint8)
        stream.synchronize()
        ptr = tensor.data_ptr()
        segment = next(s for s in torch.npu.memory._snapshot()["segments"]
                       if s["address"] <= ptr < s["address"] + s["total_size"])
        if "is_expandable" not in segment:
            raise RuntimeError("Runtime snapshot cannot verify expandable segment state")
        records.append(dict(requested=enabled, actual=segment["is_expandable"]))
        del tensor
        torch.npu.synchronize()
        torch.npu.empty_cache()
    print(json.dumps(dict(helper=args.helper, switches=records), sort_keys=True))
    if any(r["requested"] != r["actual"] for r in records):
        raise RuntimeError("Allocator helper did not change actual NPU allocation behavior")


if __name__ == "__main__":
    main()
