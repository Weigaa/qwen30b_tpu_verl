import torch
import torch_npu
import time

device = "npu:0"
dtype = torch.bfloat16  # 或 torch.float16

M_total, K, N = 4096, 2048, 1536

# 两个输入：前 6 行相同，尾部 2 行不同
x_base = torch.randn(M_total, K, device=device, dtype=dtype)

xA = x_base.clone()
xB = x_base.clone()

# 尾部行做极端扰动（两种你任选一种）
xA[M_total-2:, :] = 0
xB[M_total-2:, :] = float("nan")         # 方式1：NaN 污染
# xB[6:, :] = 1e4 * torch.randn(2, K, device=device, dtype=dtype)  # 方式2：超大随机
# xB[6:, :] = torch.randn(2, K, device=device, dtype=dtype)  # 方式2：超大随机

# 两组权重（两次 matmul）
w0 = torch.randn(8, K, N, device=device, dtype=dtype)
# w1 = torch.randn(K, N, device=device, dtype=dtype)

#group_list = torch.tensor([ 16, 161, 315, 346, 462, 574, 744, 845], device=device, dtype=torch.int64)
#group_list = torch.tensor([M_total/8, 2*M_total/8, 3*M_total/8, 4*M_total/8,  5*M_total/8, 6*M_total/8, 7*M_total/8, M_total], device=device, dtype=torch.int64)
group_list = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], device=device, dtype=torch.int64)
valid_rows = int(group_list[-1].item())
_t0 = torch.npu.Event(enable_timing=True)
_t1 = torch.npu.Event(enable_timing=True)

torch.npu.synchronize()
begin = time.time()
_t0.record()
steps = 50
# split_item=2/3 表示输出为单个张量（通常仍以 list 形式返回，取 [0]）:contentReference[oaicite:1]{index=1}

experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[
        torch_npu.profiler.ExportType.Text
        ],
    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    l2_cache=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None
)

with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
        ],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
    with_modules=False,
    with_flops=False,
    experimental_config=experimental_config) as prof:
    for step in range(steps): # 训练函数
        yA = torch_npu.npu_grouped_matmul(
            [xA], [w0],
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=0,
        )[0]
        # yB = torch_npu.npu_grouped_matmul(
        #     [xB], [w0],
        #     group_list=group_list,
        #     split_item=2,
        #     group_type=0,
        #     group_list_type=0,
        # )[0]
        prof.step()

# _t1.record()
# torch.npu.synchronize()
# end = time.time()
# print("Elapsed time: %.2f ms" % ((end - begin) * 1000))
# print("npu elapsed time: %.2f ms" % (_t0.elapsed_time(_t1)) )
# # print("xa", xA)
# # print("xb", xB)
# # print("ya", yA)
# # print("yb", yB)

# print("valid_rows:", valid_rows, "M_total:", M_total)
# print("head_equal:", torch.allclose(yA[:valid_rows], yB[:valid_rows], equal_nan=True))
# print("tail_equal:", torch.allclose(yA[valid_rows:], yB[valid_rows:], equal_nan=True))
# print("tail_has_nan:", torch.isnan(yB[valid_rows:]).any().item())
