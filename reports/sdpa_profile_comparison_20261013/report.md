# Torch SDPA 与手写 SDPA：融合优势与采集波动

## 结论

- 本次 Torch SDPA 使用 CUTLASS memory-efficient 融合注意力。每次 SDPA 调用有 **3 个 GPU kernel**；手写版本有 **9 个 kernel + 1 次 memset**。手写版本将矩阵乘法、缩放、掩码、softmax 分开执行。
- 排除预热后，Torch SDPA 的 GPU 投影均值是 **101.835 μs**。较早手写采集为 **289.875 μs**，观察到 **2.85 倍耗时差**；这仍是不同时间采集的描述性比较，最终性能倍数需要同一空闲 GPU 上复测。
- 最新手写采集均值为 **694.639 μs**，中位数 **289.044 μs**。其 FFN 均值也从约 730 μs 上升到 1559 μs，说明本次额外 GPU 延迟影响了多个模块。**不能把最新 SDPA 均值比 6.82 倍直接归结为实现差异。**

## 排除预热后的模块耗时

下表每行包含 120 次正式调用，即 10 次前向传播 × 12 层。GPU 投影覆盖该 NVTX 范围内第一个 GPU 操作开始至最后一个结束，包含中间间隙；CPU 范围耗时主要记录调用和提交过程，两者可以异步重叠。

FFN 作为对照：较早手写报告与 Torch 报告的 FFN GPU 均值只相差约 0.45%，而本次手写报告的 FFN 均值约为 Torch 的 2.15 倍。FFN 的实现没有随着 SDPA 切换而改变，这使“SDPA 代码单独造成全局翻倍”缺乏支持。

中位数能展示典型调用，但不能替代均值或证明环境没有干扰。本报告保留全部正式调用，包括长尾，不按耗时删样本。

| 版本 | 模块 | GPU均值_μs | GPU中位数_μs | GPU总体标准差_μs | GPU最大值_μs | GPU操作数 | kernel数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Torch SDPA | SDPA | 101.835 | 97.425 | 8.966 | 136.865 | 3 | 3 |
| Torch SDPA | FFN | 726.425 | 726.534 | 1.908 | 732.102 | 8 | 7 |
| 手写 · 本次 | SDPA | 694.639 | 289.044 | 923.733 | 2830.472 | 10 | 9 |
| 手写 · 本次 | FFN | 1559.037 | 733.002 | 1183.212 | 3290.415 | 8 | 7 |
| 手写 · 较早 | SDPA | 289.875 | 287.634 | 17.479 | 477.891 | 10 | 9 |
| 手写 · 较早 | FFN | 729.674 | 729.621 | 4.254 | 770.150 | 8 | 7 |

## kernel 构成解释了融合实现的优势

Torch 范围内的主 kernel 是 `fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<...>::Params)`，另有两个张量填充 kernel。主融合 kernel 的正式调用平均耗时为 **95.396 μs**。这是本次自动选择的 memory-efficient 后端，不能把名字中的 `sm80` 当作实际 GPU 型号，也不能据此认定它使用了 FlashAttention 或 Tensor Core。

较早手写范围内，两个矩阵乘法 kernel 合计约 **105.894 μs/调用**，softmax 约 **39.962 μs**，掩码填充约 **44.029 μs**，缩放乘法约 **35.230 μs**，另有掩码构造、取反、填充、拷贝及一次 memset。仅看两个矩阵乘法也会漏掉相当一部分注意力成本。

手写实现显式生成 `[4, 12, 512, 512]` 的 scores / probabilities。每个 FP32 张量按形状计算为 **48 MiB**，这些中间结果增加了读写和分配需求；此数是张量尺寸计算，不是实测峰值显存。融合实现减少独立操作与中间结果往返，可解释观察到的优势。实际 kernel 耗时和提交间隙见数据附件。

以下 kernel 时长按类别求和，不包含 memset 时长，也不等于 GPU 投影跨度。

| 版本 | 操作类别 | 每次调用kernel时长之和_μs | 每次调用kernel数 |
| --- | --- | --- | --- |
| Torch SDPA | 融合注意力 | 95.396 | 1.000 |
| Torch SDPA | 张量填充 | 2.011 | 2.000 |
| 手写 · 本次 | 矩阵乘法 | 301.231 | 2.000 |
| 手写 · 本次 | softmax | 40.144 | 1.000 |
| 手写 · 本次 | 因果掩码填充 | 255.169 | 1.000 |
| 手写 · 本次 | 张量拷贝 | 8.095 | 1.000 |
| 手写 · 本次 | 构造下三角掩码 | 2.704 | 1.000 |
| 手写 · 本次 | 逐元素乘法 | 35.320 | 1.000 |
| 手写 · 本次 | 张量填充 | 1.201 | 1.000 |
| 手写 · 本次 | 其他 | 1.456 | 1.000 |
| 手写 · 较早 | 矩阵乘法 | 105.894 | 2.000 |
| 手写 · 较早 | softmax | 39.962 | 1.000 |
| 手写 · 较早 | 因果掩码填充 | 44.029 | 1.000 |
| 手写 · 较早 | 张量拷贝 | 8.125 | 1.000 |
| 手写 · 较早 | 构造下三角掩码 | 2.714 | 1.000 |
| 手写 · 较早 | 逐元素乘法 | 35.230 | 1.000 |
| 手写 · 较早 | 张量填充 | 1.173 | 1.000 |
| 手写 · 较早 | 其他 | 1.454 | 1.000 |

## 本次 39.943 ms 的额外延迟主要出现在 GPU 侧

两次手写采集每步都执行 **447 个 kernel**，正式测量期间的 kernel 名称、顺序、grid 和 block 配置逐项相同。进入末尾同步前的时间接近，但末尾 `cudaDeviceSynchronize` 平均等待时间从 **6.532 ms** 增加到 **28.674 ms**。

两次 `masked_fill` kernel 的中位数都约 **44 μs**，本次却有部分调用达到 **2.584 ms**；额外延迟还出现在矩阵乘法和 FFN。分时调度或共享资源竞争是与这个模式相符的解释，但报告没有采集足够的 GPU 调度、历史时钟和其他进程活动，无法确认具体原因或进程。

本次手写程序输出的整模型前向均值为 **39.943 ms**，较早手写为 **18.336 ms**，Torch 为 **15.564 ms**。这些值包含 profiling 运行期开销，不能直接作为无 profiler 时的吞吐性能。相邻同步返回时间用于下图诊断，还含少量 Python 循环开销，略区别于脚本的计时区间。

| 版本 | 前向均值_ms | 前向标准差_ms | 同步前均值_ms | 同步等待均值_ms | 每步kernel数 |
| --- | --- | --- | --- | --- | --- |
| Torch SDPA | 15.564 | 0.022 | 8.932 | 6.636 | 363 |
| 手写 · 本次 | 39.943 | 1.185 | 11.276 | 28.674 | 447 |
| 手写 · 较早 | 18.336 | 1.244 | 11.810 | 6.532 | 447 |

## 实验配置与统计口径

- 模型：small，128,625,408 参数；`d_model=768`、`d_ff=3072`、12 层、12 头、词表 10,000。
- 输入：batch=4、context=512、head_dim=64；Q/K/V 形状由模型配置推导为 `[4, 12, 512, 64]`。因果注意力，无 dropout。
- 模式：`forward`，关闭梯度记录；本报告不包含反向传播或优化器耗时。
- 设备：`cuda:0`，NVIDIA H100 80GB HBM3；Torch 2.14.0+cu130，CUDA 13.0，FP32，matmul precision 为 highest。环境元数据由两份主报告的配套 JSON 确认；较早手写报告缺少独立 JSON，按其日志与匹配的 kernel 配置作为补充。
- 随机种子 336，5 次预热、10 次正式测量；采集 `cuda,nvtx,cublas`，CPU sampling/context switch 关闭。
- 对每个 NVTX 标签按 `Orig Start` 排序；180 次调用中排除前 60 次，保留后 120 次。单次模块统计的标准差采用总体定义 `ddof=0`，区别于部分 Nsight 汇总表的样本标准差。
- `sdpa_torch_forward` / `my_sdpa_torch_forward` 仅包围 SDPA 调用，不包含 Q/K/V 投影、RoPE 和输出投影。`ffn_forward` 范围包含 `ln2`、SwiGLU FFN 和残差相加，不能解释为纯 FFN 算子。
- 120 次调用来自 10 步 × 12 层，不能当作 120 个独立重复实验。报告仅作描述性统计，不给出显著性检验或置信区间。

统计使用官方 NVTX GPU Projection Trace CSV。通过 CUDA API correlation ID、进程和线程映射 kernel、memset、memcpy，逐条重建 GPU 投影跨度；三份报告的 1080 个范围均与官方记录完全一致。两份主报告的配置一致，stdout 正常且 stderr 为空。旧的导入失败报告没有参与分析。

## 后续如何得到更可靠的性能倍数

先用 `nvidia-smi` 选择空闲 GPU，确保两版使用相同卡、配置、输入布局、因果设置和数据类型。在同一时间段交替采集 Torch → 手写 → Torch → 手写，多轮记录整步耗时、SDPA GPU 投影与 FFN 对照。

同时运行不带 `nsys` 的 benchmark，用于评估整模型前向性能。Nsight 用于解释 kernel 构成及 GPU 延迟。若 FFN 对照发生明显漂移，应报告环境变化而非仅汇总成实现加速比。

可探索预构造/缓存因果掩码来减少重复操作，但本报告尚未对这些优化进行采集或验证。所有性能结论仅覆盖当前 small、FP32、batch=4、context=512、forward 配置。

## 来源与复现

主报告：`nsys_torch_sdpa_ffn_forward.nsys-rep`、`nsys_my_sdpa_ffn_forward.nsys-rep`。补充：上次自动保存的 `/tmp/nsys-report-2c19.nsys-rep`，已保留为 `evidence/nsys_my_sdpa_ffn_forward_earlier.nsys-rep`。

配套文件包括官方 CSV、原始 SHA-256、配套配置 JSON、`analyze_profiles.py`、`analysis.json`、`module_summary.csv`、`kernel_groups.csv`、`measured_kernels.csv`、`nvtx_ranges.csv` 和 `step_diagnostics.csv`。运行 `python3 analyze_profiles.py` 使用 `sqlite/` 下只读导出的三份数据库重新计算。

[NVIDIA GPU 投影统计定义](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#nvtx-gpu-proj-sum-nvtx-gpu-projection-summary)；[逐次 GPU 投影记录](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#nvtx-gpu-proj-trace-nvtx-gpu-projection-trace)；[PyTorch 2.14 SDPA API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)。
