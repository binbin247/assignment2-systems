# Problem (nsys_profile)：当前配置的答案草稿

范围：small，batch=4，context=512，FP32，NVIDIA H100 80GB；5 次预热、10 次正式测量。以下为已采集配置的回答，仍需补齐第二种模型规模和三个符合题目要求的上下文长度。

## (a)

在 Nsight 时间线上，排除预热后的整模型前向 GPU 跨度平均为 **15.488 ms**，此前未开启 nsys 的 Python `timeit.default_timer()` 基准为 **15.321 ms**，相差约 **1.1%**，结果基本一致。开启 profiling 后，同一次运行中 Python 脚本记录的整步时间为 **15.564 ms**，其中还包含 CPU 提交、同步及采集开销。

## (b)

前向传播累计 GPU 时间最多的 kernel 是 `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas`，10 次正式前向共调用 **610 次**，即每次前向 **61 次**，累计 **65.548 ms**。在已有完整训练采集中，同一 kernel 仍最多（10 步累计 **65.589 ms**）；AdamW 的更新由逐元素操作组成，因此加入反向后，主导的单个 GEMM kernel 仍是它。

## (c)

除矩阵乘法外，还观察到了逐元素乘法、加法/残差、SiLU 和张量复制等 kernel，在手写注意力实现中还存在独立的 softmax 和因果掩码填充。较早手写采集中，softmax 与 masked_fill 每次注意力调用分别耗时约 **39.962 μs** 和 **44.029 μs**，属于不可忽略的额外成本。

## (d)

按“独立 GEMM/SGEMM kernel 时长占所有 GPU kernel 累计时长”统计，矩阵乘法占比由推理的 **82.1%** 降至完整 AdamW 训练的 **69.6%**，其余 kernel 的占比由 **17.9%** 升至 **30.4%**。训练增加了损失计算、梯度计算与 AdamW 状态/参数更新，引入更多逐元素和归约操作；融合注意力 kernel 含有矩阵计算但在此统计中单列，不能把这些比例当作全部数学矩阵乘法的精确占比。

## (e)

在较早手写 SDPA 的前向采集中，softmax 每次耗时约 **39.962 μs**，`QKᵀ` 与 `PV` 两次矩阵乘法合计约 **105.894 μs**，矩阵乘法耗时约为 softmax 的 **2.65 倍**。两个矩阵乘法的运算量约为 `4BHT²d_k`（本配置约 **3.22 GFLOPs**），而 softmax 的逐元素与归约运算量按 `BHT²` 增长；计算量差距明显大于耗时差距，说明内存读写、归约和指数运算成本也会影响实际耗时。

## 统计说明

- 前向 GPU 跨度取每步首个 GPU 操作开始至最后一个结束，包含 GPU 操作间隙；它与“所有 kernel 时长之和”是不同指标。
- 所有 kernel 排名和占比均排除前 5 步预热，使用后 10 步。训练的最后一个 profiler 清理同步不包含模型 kernel，未计入正式步骤。
- 训练报告额外开启了 autograd NVTX，以上比例是不同采集中的观察结果。
- (e) 使用手写版本以便分离 softmax 和 GEMM；Torch SDPA 的融合注意力 kernel 不能直接拆出独立 softmax 时间。
- 未使用最新手写报告中的异常长尾来估算正常实现性能。
- 来源：`nsys_torch_sdpa_ffn_forward.nsys-rep`、`nsys_walkthrough.nsys-rep`、较早手写采集及 `benchmarking_results.json`。完整统计保存在 `nsys_profile_answer_evidence.json`。
