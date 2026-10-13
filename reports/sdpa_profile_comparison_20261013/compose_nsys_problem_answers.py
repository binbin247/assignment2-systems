"""Create scoped answers to nsys_profile from already-recorded captures."""
import collections
import json
import sqlite3
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent

def measured_capture(path):
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    sync=[dict(r) for r in db.execute("SELECT r.start,r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE 'cudaDeviceSynchronize%' ORDER BY r.start")]
    kernels=[dict(r) for r in db.execute('SELECT k.start,k.end,s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start')]
    ops=[(r['start'],r['end']) for r in kernels]
    for table in ['CUPTI_ACTIVITY_KIND_MEMSET','CUPTI_ACTIVITY_KIND_MEMCPY']:
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone():
            ops.extend((r['start'],r['end']) for r in db.execute('SELECT start,end FROM '+table))
    intervals=[]
    for previous,current in zip(sync[:-1],sync[1:]):
        ks=[r for r in kernels if previous['end']<=r['start'] and r['end']<=current['end']]
        gpu=[r for r in ops if previous['end']<=r[0] and r[1]<=current['end']]
        if ks:intervals.append({'kernels':ks,'span_ms':(max(r[1] for r in gpu)-min(r[0] for r in gpu))/1e6})
    # The profiler's final emit_nvtx cleanup sync has no model kernels; omit it.
    selected=intervals[-10:]
    assert len(selected)==10
    rows=[r for step in selected for r in step['kernels']]
    grouped=collections.defaultdict(list)
    for r in rows:grouped[r['name']].append((r['end']-r['start'])/1e6)
    ranking=sorted([{'name':name,'calls':len(values),'calls_per_step':len(values)/10,'total_ms':sum(values)} for name,values in grouped.items()],key=lambda r:r['total_ms'],reverse=True)
    total=sum(r['end']-r['start'] for r in rows)/1e6
    gemm=sum(r['end']-r['start'] for r in rows if 'gemm' in r['name'].lower())/1e6
    answer={'gpu_span_ms_mean':statistics.mean(step['span_ms'] for step in selected),
            'kernel_count_per_step':[len(step['kernels']) for step in selected],
            'total_kernel_ms':total,'gemm_kernel_ms':gemm,'gemm_fraction':gemm/total,
            'non_gemm_fraction':1-gemm/total,'kernel_ranking':ranking}
    db.close()
    return answer

forward=measured_capture(ROOT/'sqlite/torch.sqlite')
train=measured_capture(ROOT/'sqlite/train_walkthrough.sqlite')
benchmark=json.loads((ROOT/'answer_baseline_forward.json').read_text())
baseline=benchmark['summary']['total']['mean_ms']
a=json.loads((ROOT/'analysis.json').read_text())
softmax=next(r for r in a['kernel_groups'] if r['capture']=='my_earlier' and r['module']=='SDPA' and r['category']=='softmax')['sum_per_call_us']
gemm=sum(r['sum_per_call_us'] for r in a['kernel_groups'] if r['capture']=='my_earlier' and r['module']=='SDPA' and r['category']=='矩阵乘法')
mask=next(r for r in a['kernel_groups'] if r['capture']=='my_earlier' and r['module']=='SDPA' and r['category']=='因果掩码填充')['sum_per_call_us']
assert forward['kernel_ranking'][0]['name']==train['kernel_ranking'][0]['name']
facts={'scope':{'model':'small','batch':4,'context':512,'dtype':'float32','gpu':'H100 80GB'},
       'baseline_forward_ms':baseline,'baseline_std_ms':benchmark['summary']['total']['std_ms'],
       'forward':forward,'train':train,'softmax_us':softmax,'sdpa_two_gemm_us':gemm,
       'gpu_scope':'GEMM fractions use sum of kernel durations; fused attention is a separate class; gaps and memory operations excluded.',
       'limitations':['Only small/context=512 covered; not the full two-size/three-length assignment.',
                      'Training capture additionally enabled autograd NVTX.',
                      'Part e uses earlier handwritten SDPA capture to expose separate softmax and GEMM kernels.']}
(ROOT/'nsys_profile_answer_evidence.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2))
full_kernel=forward['kernel_ranking'][0]['name']
text=f'''# Problem (nsys_profile)：当前配置的答案草稿

范围：small，batch=4，context=512，FP32，NVIDIA H100 80GB；5 次预热、10 次正式测量。以下为已采集配置的回答，仍需补齐第二种模型规模和三个符合题目要求的上下文长度。

## (a)

在 Nsight 时间线上，排除预热后的整模型前向 GPU 跨度平均为 **{forward['gpu_span_ms_mean']:.3f} ms**，此前未开启 nsys 的 Python `timeit.default_timer()` 基准为 **{baseline:.3f} ms**，相差约 **{(forward['gpu_span_ms_mean']/baseline-1)*100:.1f}%**，结果基本一致。开启 profiling 后，同一次运行中 Python 脚本记录的整步时间为 **15.564 ms**，其中还包含 CPU 提交、同步及采集开销。

## (b)

前向传播累计 GPU 时间最多的 kernel 是 `{full_kernel}`，10 次正式前向共调用 **{forward['kernel_ranking'][0]['calls']} 次**，即每次前向 **{forward['kernel_ranking'][0]['calls_per_step']:.0f} 次**，累计 **{forward['kernel_ranking'][0]['total_ms']:.3f} ms**。在已有完整训练采集中，同一 kernel 仍最多（10 步累计 **{train['kernel_ranking'][0]['total_ms']:.3f} ms**）；AdamW 的更新由逐元素操作组成，因此加入反向后，主导的单个 GEMM kernel 仍是它。

## (c)

除矩阵乘法外，还观察到了逐元素乘法、加法/残差、SiLU 和张量复制等 kernel，在手写注意力实现中还存在独立的 softmax 和因果掩码填充。较早手写采集中，softmax 与 masked_fill 每次注意力调用分别耗时约 **{softmax:.3f} μs** 和 **{mask:.3f} μs**，属于不可忽略的额外成本。

## (d)

按“独立 GEMM/SGEMM kernel 时长占所有 GPU kernel 累计时长”统计，矩阵乘法占比由推理的 **{forward['gemm_fraction']*100:.1f}%** 降至完整 AdamW 训练的 **{train['gemm_fraction']*100:.1f}%**，其余 kernel 的占比由 **{forward['non_gemm_fraction']*100:.1f}%** 升至 **{train['non_gemm_fraction']*100:.1f}%**。训练增加了损失计算、梯度计算与 AdamW 状态/参数更新，引入更多逐元素和归约操作；融合注意力 kernel 含有矩阵计算但在此统计中单列，不能把这些比例当作全部数学矩阵乘法的精确占比。

## (e)

在较早手写 SDPA 的前向采集中，softmax 每次耗时约 **{softmax:.3f} μs**，`QKᵀ` 与 `PV` 两次矩阵乘法合计约 **{gemm:.3f} μs**，矩阵乘法耗时约为 softmax 的 **{gemm/softmax:.2f} 倍**。两个矩阵乘法的运算量约为 `4BHT²d_k`（本配置约 **3.22 GFLOPs**），而 softmax 的逐元素与归约运算量按 `BHT²` 增长；计算量差距明显大于耗时差距，说明内存读写、归约和指数运算成本也会影响实际耗时。

## 统计说明

- 前向 GPU 跨度取每步首个 GPU 操作开始至最后一个结束，包含 GPU 操作间隙；它与“所有 kernel 时长之和”是不同指标。
- 所有 kernel 排名和占比均排除前 5 步预热，使用后 10 步。训练的最后一个 profiler 清理同步不包含模型 kernel，未计入正式步骤。
- 训练报告额外开启了 autograd NVTX，以上比例是不同采集中的观察结果。
- (e) 使用手写版本以便分离 softmax 和 GEMM；Torch SDPA 的融合注意力 kernel 不能直接拆出独立 softmax 时间。
- 未使用最新手写报告中的异常长尾来估算正常实现性能。
- 来源：`nsys_torch_sdpa_ffn_forward.nsys-rep`、`nsys_walkthrough.nsys-rep`、较早手写采集及 `benchmarking_results.json`。完整统计保存在 `nsys_profile_answer_evidence.json`。
'''
(ROOT/'nsys_profile_answers_small512.md').write_text(text)
print(json.dumps({'forward_gpu_ms':forward['gpu_span_ms_mean'],'baseline_ms':baseline,
                  'gemm_inference_fraction':forward['gemm_fraction'],'gemm_training_fraction':train['gemm_fraction'],
                  'top_kernel_calls_per_step':forward['kernel_ranking'][0]['calls_per_step'],
                  'non_matmul_examples':[r['name'][:120] for r in forward['kernel_ranking'] if any(k in r['name'].lower() for k in ['silu','reduce','rms','welford'])]},ensure_ascii=False))
