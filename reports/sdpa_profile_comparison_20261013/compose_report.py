"""Compose the evidence snapshot and the text companion from analysis.json."""
import collections
import datetime
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
a=json.loads((ROOT/'analysis.json').read_text())
s={(r['capture'],r['module']):r for r in a['module_summary']}
t=s['torch','SDPA']; old=s['my_earlier','SDPA']; new=s['my_latest','SDPA']
names={'torch':'Torch SDPA','my_latest':'手写 · 本次','my_earlier':'手写 · 较早'}
summary_rows=[]
for r in a['module_summary']:
    summary_rows.append({'版本':names[r['capture']],'模块':r['module'],'GPU均值_μs':r['gpu_projected_us_mean'],
                         'GPU中位数_μs':r['gpu_projected_us_median'],'GPU总体标准差_μs':r['gpu_projected_us_std_population'],
                         'GPU最小值_μs':r['gpu_projected_us_min'],'GPU最大值_μs':r['gpu_projected_us_max'],
                         'CPU范围均值_μs':r['cpu_range_us_mean'],'GPU操作数':r['gpu_ops_per_call'],
                         'kernel数':r['kernels_per_call'],'正式调用数':120})
step_rows=[]
for cap in names:
    rows=[r for r in a['step_diagnostics'] if r['capture']==cap]
    c=next(c for c in a['captures'] if c['capture']==cap)
    step_rows.append({'版本':names[cap],'前向均值_ms':c['benchmark_mean_ms'],'前向标准差_ms':c['benchmark_std_ms'],
                      '同步前均值_ms':sum(r['before_sync_ms'] for r in rows)/10,
                      '同步等待均值_ms':sum(r['sync_wait_ms'] for r in rows)/10,
                      '每步kernel数':rows[0]['kernel_count']})
kernel_rows=[]
for cap in names:
    buckets=collections.defaultdict(list)
    for k in a['kernel_groups']:
        if k['capture']==cap and k['module']=='SDPA':buckets[k['category']].append(k)
    for category,items in buckets.items():
        kernel_rows.append({'版本':names[cap],'操作类别':category,'每次调用kernel时长之和_μs':sum(k['sum_per_call_us'] for k in items),
                            '每次调用kernel数':sum(k['instances'] for k in items)/120})
ranges=[{'版本':names[r['capture']],'模块':r['module'],'正式调用':r['call']-60,'步骤':r['step']-5,
         '层':r['layer'],'GPU投影耗时_μs':r['gpu_projected_us'],'CPU范围耗时_μs':r['cpu_range_us']}
        for r in a['nvtx_ranges'] if not r['warmup']]
step_chart=[{'版本':names[r['capture']],'步骤':r['step'],'同步等待_ms':r['sync_wait_ms'],
             '同步返回间隔_ms':r['sync_interval_ms']} for r in a['step_diagnostics']]

executive=f'''## 结论

- 本次 Torch SDPA 使用 CUTLASS memory-efficient 融合注意力。每次 SDPA 调用有 **3 个 GPU kernel**；手写版本有 **9 个 kernel + 1 次 memset**。手写版本将矩阵乘法、缩放、掩码、softmax 分开执行。
- 排除预热后，Torch SDPA 的 GPU 投影均值是 **{t['gpu_projected_us_mean']:.3f} μs**。较早手写采集为 **{old['gpu_projected_us_mean']:.3f} μs**，观察到 **{old['gpu_projected_us_mean']/t['gpu_projected_us_mean']:.2f} 倍耗时差**；这仍是不同时间采集的描述性比较，最终性能倍数需要同一空闲 GPU 上复测。
- 最新手写采集均值为 **{new['gpu_projected_us_mean']:.3f} μs**，中位数 **{new['gpu_projected_us_median']:.3f} μs**。其 FFN 均值也从约 730 μs 上升到 1559 μs，说明本次额外 GPU 延迟影响了多个模块。**不能把最新 SDPA 均值比 {new['gpu_projected_us_mean']/t['gpu_projected_us_mean']:.2f} 倍直接归结为实现差异。**'''
timing='''## 排除预热后的模块耗时

下表每行包含 120 次正式调用，即 10 次前向传播 × 12 层。GPU 投影覆盖该 NVTX 范围内第一个 GPU 操作开始至最后一个结束，包含中间间隙；CPU 范围耗时主要记录调用和提交过程，两者可以异步重叠。

FFN 作为对照：较早手写报告与 Torch 报告的 FFN GPU 均值只相差约 0.45%，而本次手写报告的 FFN 均值约为 Torch 的 2.15 倍。FFN 的实现没有随着 SDPA 切换而改变，这使“SDPA 代码单独造成全局翻倍”缺乏支持。

中位数能展示典型调用，但不能替代均值或证明环境没有干扰。本报告保留全部正式调用，包括长尾，不按耗时删样本。'''
kernels='''## kernel 构成解释了融合实现的优势

Torch 范围内的主 kernel 是 `fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<...>::Params)`，另有两个张量填充 kernel。主融合 kernel 的正式调用平均耗时为 **95.396 μs**。这是本次自动选择的 memory-efficient 后端，不能把名字中的 `sm80` 当作实际 GPU 型号，也不能据此认定它使用了 FlashAttention 或 Tensor Core。

较早手写范围内，两个矩阵乘法 kernel 合计约 **105.894 μs/调用**，softmax 约 **39.962 μs**，掩码填充约 **44.029 μs**，缩放乘法约 **35.230 μs**，另有掩码构造、取反、填充、拷贝及一次 memset。仅看两个矩阵乘法也会漏掉相当一部分注意力成本。

手写实现显式生成 `[4, 12, 512, 512]` 的 scores / probabilities。每个 FP32 张量按形状计算为 **48 MiB**，这些中间结果增加了读写和分配需求；此数是张量尺寸计算，不是实测峰值显存。融合实现减少独立操作与中间结果往返，可解释观察到的优势。实际 kernel 耗时和提交间隙见数据附件。

以下 kernel 时长按类别求和，不包含 memset 时长，也不等于 GPU 投影跨度。'''
variance='''## 本次 39.943 ms 的额外延迟主要出现在 GPU 侧

两次手写采集每步都执行 **447 个 kernel**，正式测量期间的 kernel 名称、顺序、grid 和 block 配置逐项相同。进入末尾同步前的时间接近，但末尾 `cudaDeviceSynchronize` 平均等待时间从 **6.532 ms** 增加到 **28.674 ms**。

两次 `masked_fill` kernel 的中位数都约 **44 μs**，本次却有部分调用达到 **2.584 ms**；额外延迟还出现在矩阵乘法和 FFN。分时调度或共享资源竞争是与这个模式相符的解释，但报告没有采集足够的 GPU 调度、历史时钟和其他进程活动，无法确认具体原因或进程。

本次手写程序输出的整模型前向均值为 **39.943 ms**，较早手写为 **18.336 ms**，Torch 为 **15.564 ms**。这些值包含 profiling 运行期开销，不能直接作为无 profiler 时的吞吐性能。相邻同步返回时间用于下图诊断，还含少量 Python 循环开销，略区别于脚本的计时区间。'''
method='''## 实验配置与统计口径

- 模型：small，128,625,408 参数；`d_model=768`、`d_ff=3072`、12 层、12 头、词表 10,000。
- 输入：batch=4、context=512、head_dim=64；Q/K/V 形状由模型配置推导为 `[4, 12, 512, 64]`。因果注意力，无 dropout。
- 模式：`forward`，关闭梯度记录；本报告不包含反向传播或优化器耗时。
- 设备：`cuda:0`，NVIDIA H100 80GB HBM3；Torch 2.14.0+cu130，CUDA 13.0，FP32，matmul precision 为 highest。环境元数据由两份主报告的配套 JSON 确认；较早手写报告缺少独立 JSON，按其日志与匹配的 kernel 配置作为补充。
- 随机种子 336，5 次预热、10 次正式测量；采集 `cuda,nvtx,cublas`，CPU sampling/context switch 关闭。
- 对每个 NVTX 标签按 `Orig Start` 排序；180 次调用中排除前 60 次，保留后 120 次。单次模块统计的标准差采用总体定义 `ddof=0`，区别于部分 Nsight 汇总表的样本标准差。
- `sdpa_torch_forward` / `my_sdpa_torch_forward` 仅包围 SDPA 调用，不包含 Q/K/V 投影、RoPE 和输出投影。`ffn_forward` 范围包含 `ln2`、SwiGLU FFN 和残差相加，不能解释为纯 FFN 算子。
- 120 次调用来自 10 步 × 12 层，不能当作 120 个独立重复实验。报告仅作描述性统计，不给出显著性检验或置信区间。

统计使用官方 NVTX GPU Projection Trace CSV。通过 CUDA API correlation ID、进程和线程映射 kernel、memset、memcpy，逐条重建 GPU 投影跨度；三份报告的 1080 个范围均与官方记录完全一致。两份主报告的配置一致，stdout 正常且 stderr 为空。旧的导入失败报告没有参与分析。'''
next_steps='''## 后续如何得到更可靠的性能倍数

先用 `nvidia-smi` 选择空闲 GPU，确保两版使用相同卡、配置、输入布局、因果设置和数据类型。在同一时间段交替采集 Torch → 手写 → Torch → 手写，多轮记录整步耗时、SDPA GPU 投影与 FFN 对照。

同时运行不带 `nsys` 的 benchmark，用于评估整模型前向性能。Nsight 用于解释 kernel 构成及 GPU 延迟。若 FFN 对照发生明显漂移，应报告环境变化而非仅汇总成实现加速比。

可探索预构造/缓存因果掩码来减少重复操作，但本报告尚未对这些优化进行采集或验证。所有性能结论仅覆盖当前 small、FP32、batch=4、context=512、forward 配置。'''
sources='''## 来源与复现

主报告：`nsys_torch_sdpa_ffn_forward.nsys-rep`、`nsys_my_sdpa_ffn_forward.nsys-rep`。补充：上次自动保存的 `/tmp/nsys-report-2c19.nsys-rep`，已保留为 `evidence/nsys_my_sdpa_ffn_forward_earlier.nsys-rep`。

配套文件包括官方 CSV、原始 SHA-256、配套配置 JSON、`analyze_profiles.py`、`analysis.json`、`module_summary.csv`、`kernel_groups.csv`、`measured_kernels.csv`、`nvtx_ranges.csv` 和 `step_diagnostics.csv`。运行 `python3 analyze_profiles.py` 使用 `sqlite/` 下只读导出的三份数据库重新计算。

[NVIDIA GPU 投影统计定义](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#nvtx-gpu-proj-sum-nvtx-gpu-projection-summary)；[逐次 GPU 投影记录](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#nvtx-gpu-proj-trace-nvtx-gpu-projection-trace)；[PyTorch 2.14 SDPA API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)。'''
sections={'executive':executive,'timing':timing,'kernels':kernels,'variance':variance,'method':method,'next_steps':next_steps,'sources':sources}
def table(rows,cols):
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(f'{r[c]:.3f}' if isinstance(r[c],float) else str(r[c]) for c in cols)+' |' for r in rows])
markdown='# Torch SDPA 与手写 SDPA：融合优势与采集波动\n\n'+executive+'\n\n'+timing+'\n\n'+table(summary_rows,['版本','模块','GPU均值_μs','GPU中位数_μs','GPU总体标准差_μs','GPU最大值_μs','GPU操作数','kernel数'])+'\n\n'+kernels+'\n\n'+table(kernel_rows,['版本','操作类别','每次调用kernel时长之和_μs','每次调用kernel数'])+'\n\n'+variance+'\n\n'+table(step_rows,list(step_rows[0]))+'\n\n'+method+'\n\n'+next_steps+'\n\n'+sources+'\n'
(ROOT/'report.md').write_text(markdown)
(ROOT/'report_content.json').write_text(json.dumps(sections,ensure_ascii=False,indent=2))
docs=[{'label':'NVIDIA NVTX GPU Projection','url':'https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#nvtx-gpu-proj-sum-nvtx-gpu-projection-summary'},
      {'label':'PyTorch 2.14 SDPA','url':'https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html'}]
def source(label,files,component_ids,definition):
    return {'label':label,'provider':'NVIDIA Nsight Systems 2025.5.2','files':files,
            'tables':['NVTX_EVENTS','CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_KERNEL'],
            'filters':['forward; FP32; H100; B=4; T=512; 12 layers; discard first 60 of 180 instances per NVTX label'],
            'links':docs,'caveats':['Latest handwritten capture has extra GPU latency; speed ratios are descriptive, not causal.'],
            'metricDefinitions':[{'label':label,'definition':definition,'componentIds':component_ids,'sourceLineage':[{'files':files}]}],
            'evidenceFlow':[{'title':'实际采集','detail':'Original captures matched by SHA-256 in evidence/manifest.json.'},
                            {'title':'官方导出','detail':'nsys stats --report nvtx_gpu_proj_trace --format csv <capture.nsys-rep>'},
                            {'title':'排除预热并统计','detail':'Sort each NVTX label by Orig Start; discard 60, retain 120; mean/median/pstdev computed in analyze_profiles.py.'}]}
files=[c['capture']+'_nvtx_gpu_proj_trace.csv' for c in a['captures']]
queries={
 'module_summary':{'rows':summary_rows,'source':source('模块 GPU 投影统计',files,['executive','timing','module-table','sdpa-timing','ffn-timing'],'First enclosed GPU operation start through last enclosed GPU operation end, including gaps. Units μs; 120 measured calls; population std ddof=0.')},
 'call_trace':{'rows':ranges,'source':source('正式调用的 GPU 投影逐次记录',files,['sdpa-calls'],'All retained 120 calls per label and capture; values taken directly from official projected durations in ns.')},
 'kernel_components':{'rows':kernel_rows,'source':source('SDPA kernel 分类时长', ['measured_kernels.csv'],['kernels','kernel-table'],'Sum CUDA kernel durations by category divided by 120. Excludes memset and inter-kernel gaps; does not equal GPU projected span.'),'methods':[{'language':'sql','code':a['method']['sql_kernel_mapping']}]},
 'step_summary':{'rows':step_rows,'source':source('整步耗时与同步等待', ['step_diagnostics.csv','manifest.json'],['variance','step-table'],'Benchmark total mean from capture stdout; CUDA DeviceSynchronize wait from runtime API timestamps. All values ms.')},
 'step_trace':{'rows':step_chart,'source':source('末尾同步逐步等待', ['step_diagnostics.csv'],['wait-steps'],'Duration of final cudaDeviceSynchronize for each of last 10 steps, in ms. Synchronization waits for previously submitted GPU work.')},
 'capture_metadata':{'rows':[{'版本':names[c['capture']],'采集UTC':c['capture_utc'],'原始报告字节数':c['report_bytes'],'SHA256':c['sha256']} for c in a['captures']],
                     'source':source('原始采集来源', ['manifest.json'],['method','source-table'],'Recorded original report hashes and Nsight session timestamps; immutable capture evidence.')}
}
snapshot={'surface':'report','title':'Torch SDPA 与手写 SDPA：融合优势与采集波动','generatedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'status':'reviewed','buildStatus':'creating','filters':[],'queries':queries,
          'report':{'asOf':a['captures'][1]['capture_utc']+'Z'}}
(ROOT/'reviewed.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
print('Composed report.md, reviewed.json and report_content.json')
