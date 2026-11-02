# Multi-stage DP Sidecars

`internal/watch_elastic_shrink_and_run_multistage_sidecars.sh` consumes every
new rank release in a rollout window. With a 16-rank world and target floors
`8,4,2`, the expected launches are as follows.

| Transition | Newly released devices | Sidecar layout |
| --- | --- | --- |
| 16 to 8 | 8 through 15 | Eight independent TP1 replicas |
| 8 to 4 | 4 through 7 | Four independent TP1 replicas |
| 4 to 2 | 2 and 3 | Two independent TP1 replicas |

Earlier sidecars remain alive when a deeper shrink releases more devices. A
rollout completion or restore marker sends a soft-stop request to every stage
and then terminates any process that misses the restore deadline. The watcher
resets after restore, so the same behavior is available in later training
steps.

The released physical rank is also the global prompt shard id. All stages use
16 global shards and a common checkpoint directory. This prevents later
sidecars from duplicating work already assigned to an earlier release stage.

Qwen2.5-1.5B-Instruct can be prepared without starting inference by running

```bash
./prepare_qwen2_5_1_5b_sidecar_assets.sh
```

After the model is present, the full AdaFloor entry point is

```bash
./run_adafloor_natural_floor2_with_qwen2_5_1_5b_multistage_dp_sidecars.sh
```

The wrapper keeps `TP=1`, disables expert parallelism, and launches one
independent vLLM process per newly released NPU.
