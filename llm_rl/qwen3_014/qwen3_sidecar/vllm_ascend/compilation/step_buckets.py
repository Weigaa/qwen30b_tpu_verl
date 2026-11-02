"""Update all v0.14 dispatch/capture shape state at a sleeping step boundary."""


def configure_step_buckets(runner, sizes):
    normalized = [int(x) for x in sizes]
    if not normalized or normalized != sorted(set(normalized)) or normalized[0] < 1 or normalized[-1] != 32:
        raise ValueError(f"Invalid frozen step buckets: {sizes}")
    config = runner.compilation_config
    if normalized == list(config.cudagraph_capture_sizes):
        return False
    from vllm_ascend.compilation.acl_graph import (
        disable_aclgraph_dispatch, reconfigure_graph_params)
    # Sleep lifecycle has already retired old graphs before this host-only
    # shape update. Keys and padding lookup must match the new capture list.
    disable_aclgraph_dispatch(runner)
    config.cudagraph_capture_sizes = normalized
    config.max_cudagraph_capture_size = normalized[-1]
    config.compute_bs_to_padded_graph_size()
    runner.cudagraph_batch_sizes = normalized
    # KV-first wake reinitializes Attention BEFORE capture_model. Its
    # set_graph_params validates this table against runner sizes, so replace
    # the retired epoch's task tables now, not only during later capture.
    reconfigure_graph_params(set(normalized))
    return True
