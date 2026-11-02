"""Per-instance profiling with an independent physical-HBM safety bound."""


def instance_kv_budget(*,total_bytes,utilization,baseline_non_torch_bytes,
                       peak_torch_bytes,current_torch_bytes,
                       global_non_torch_bytes,physical_free_bytes):
    """Do not charge a parked primary's pre-existing allocation to vLLM twice.

    CacheConfig.gpu_memory_utilization is a per-instance limit. The legacy
    Ascend profiler counts all other processes as this instance's non-Torch
    allocation. Subtract the before-load baseline only from that attribution,
    never from physical usage. Physical capacity still reserves the observed
    transient Torch peak and at least1GiB/2% HBM for graph/runtime growth.
    The primary must remain parked until the sidecar's physical-exit ACK.
    """
    if not 0<utilization<=1:raise ValueError('Invalid per-instance fraction')
    own_non_torch=max(0,global_non_torch_bytes-baseline_non_torch_bytes)
    own_peak=peak_torch_bytes+own_non_torch
    instance_limit=int(total_bytes*utilization)
    instance_kv_limit=max(0,instance_limit-own_peak)
    transient=max(0,peak_torch_bytes-current_torch_bytes)
    physical_headroom=max(1024**3,int(total_bytes*.02))
    physical_kv_limit=max(0,physical_free_bytes-transient-physical_headroom)
    chosen=int(min(instance_kv_limit,physical_kv_limit))
    return dict(total_bytes=total_bytes,utilization=utilization,
        baseline_non_torch_bytes=baseline_non_torch_bytes,
        global_non_torch_bytes=global_non_torch_bytes,own_non_torch_bytes=own_non_torch,
        peak_torch_bytes=peak_torch_bytes,current_torch_bytes=current_torch_bytes,
        own_peak_bytes=own_peak,instance_limit_bytes=instance_limit,
        instance_kv_limit_bytes=instance_kv_limit,physical_free_bytes=physical_free_bytes,
        transient_headroom_bytes=transient,physical_headroom_bytes=physical_headroom,
        physical_kv_limit_bytes=physical_kv_limit,kv_budget_bytes=chosen)
