"""Match pasted Nsight tables to official exports and remove warmup calls."""
import collections
import csv
import hashlib
import json
import re
import statistics
from decimal import Decimal
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'step2';OUT.mkdir(exist_ok=True)
SOURCES=[
 ('my_latest','手写 SDPA',OUT/'my_latest_pasted.tsv'),
 ('torch','Torch SDPA',OUT/'torch_pasted.tsv'),
]
FACTORS={'s':Decimal(10)**9,'ms':Decimal(10)**6,'μs':Decimal(10)**3,'µs':Decimal(10)**3,'us':Decimal(10)**3,'ns':Decimal(1)}

def displayed_ns(text):
    match=re.fullmatch(r'\s*([\d.,]+)\s*(ms|μs|µs|us|ns|s)\s*',text)
    assert match,text
    number=Decimal(match[1].replace(',',''));factor=FACTORS[match[2]]
    # Displayed values may omit trailing zeros; this gives a conservative bound.
    quantum=Decimal(10)**number.as_tuple().exponent*factor
    return number*factor,quantum/2+Decimal(1)

def export(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

retained=[];excluded=[];summaries=[];validation=[]
for capture,label,filename in SOURCES:
    source=Path(filename)
    with source.open(newline='',encoding='utf-8') as f: pasted=list(csv.DictReader(f,delimiter='\t'))
    with (ROOT/'evidence'/(capture+'_nvtx_gpu_proj_trace.csv')).open(newline='',encoding='utf-8') as f: official=list(csv.DictReader(f))
    assert len(pasted)==len(official)==360
    official_by_id={r['RangeId']:r for r in official}
    assert len(official_by_id)==360
    assert {r['RangeId'] for r in pasted}==set(official_by_id)
    for r in pasted:
        actual=official_by_id[r['RangeId']]
        for key in ['Name','Style','PID','TID','NumGPUOps']:
            assert r[key]==actual[key],(label,r['RangeId'],key)
        for key in ['Projected Start','Projected Duration','Orig Start','Orig Duration']:
            value,tolerance=displayed_ns(r[key]);expected=Decimal(actual[key+' (ns)'])
            assert abs(value-expected)<=tolerance,(label,r['RangeId'],key,value,expected,tolerance)
    by_name=collections.defaultdict(list)
    for r in pasted:by_name[r['Name']].append(r)
    for name,rows in by_name.items():
        rows.sort(key=lambda r:(displayed_ns(r['Orig Start'])[0],int(r['RangeId'])))
        assert len(rows)==180
        assert [r['RangeId'] for r in rows]==[r['RangeId'] for r in sorted((x for x in official if x['Name']==name),key=lambda r:int(r['Orig Start (ns)']))]
        measured=[]
        for index,p in enumerate(rows):
            r=official_by_id[p['RangeId']]
            row={'capture':capture,'version':label,'range':name,'call_index':index+1,'RangeId':int(r['RangeId']),
                 'Orig_Start_ns':int(r['Orig Start (ns)']),'Projected_Start_ns':int(r['Projected Start (ns)']),
                 'Projected_Duration_us':int(r['Projected Duration (ns)'])/1000,
                 'Orig_Duration_us':int(r['Orig Duration (ns)'])/1000,'GPU_ops':int(r['NumGPUOps']),
                 'warmup':index<60,'measured_step':None if index<60 else (index-60)//12+1,'layer':index%12+1}
            (excluded if index<60 else retained).append(row)
            if index>=60:measured.append(row)
        values=[r['Projected_Duration_us'] for r in measured]
        summaries.append({'version':label,'module':'FFN' if name.endswith('ffn_forward') else 'SDPA','range':name,
                          'original_calls':180,'excluded_warmup_calls':60,'measured_calls':120,
                          'first_measured_range_id':measured[0]['RangeId'],'last_measured_range_id':measured[-1]['RangeId'],
                          'first_measured_original_start_s':measured[0]['Orig_Start_ns']/1e9,
                          'mean_us':statistics.mean(values),'median_us':statistics.median(values),
                          'std_population_us':statistics.pstdev(values),'min_us':min(values),'max_us':max(values),
                          'gpu_ops_per_call':statistics.mean(r['GPU_ops'] for r in measured)})
    (OUT/(capture+'_pasted.tsv')).write_bytes(source.read_bytes())
    validation.append({'capture':capture,'matched_ranges':360,'all_metadata_matched':True,
                       'all_times_within_display_rounding':True,'per_label_sort_order_matched':True,
                       'pasted_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
assert len(retained)==480 and len(excluded)==240
export('measured_calls.csv',retained);export('excluded_warmup.csv',excluded);export('summary.csv',summaries)
result={'method':'Per NVTX label, sort by Orig Start; discard first 60 and retain last 120. Pasted values matched to official ns-resolution CSV by RangeId; statistics use official values.',
        'std_definition':'population standard deviation, ddof=0','retained_rows':480,'excluded_rows':240,
        'summary':summaries,'validation':validation}
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
lines=['# 第二步：剔除预热后的统计','',
       '两份粘贴表各有 360 条记录；每份按 SDPA、FFN 标签分别排序并剔除前 60 次，各保留 120 次正式调用。合计排除 240 条，保留 480 条。',
       '', '粘贴表的版本、PID/TID、RangeId 和全部时间值均与原始报告匹配。为避免界面 ms/μs 四舍五入造成偏差，以下统计采用官方导出的纳秒精度。标准差采用总体定义 ddof=0。',
       '', '| 版本 | 模块 | 正式调用数 | GPU 投影均值 μs | 中位数 μs | 总体标准差 μs | 最小值 μs | 最大值 μs |',
       '| --- | --- | --- | --- | --- | --- | --- | --- |']
for r in summaries:
    lines.append(f"| {r['version']} | {r['module']} | 120 | {r['mean_us']:.3f} | {r['median_us']:.3f} | {r['std_population_us']:.3f} | {r['min_us']:.3f} | {r['max_us']:.3f} |")
lines+=['','保留的 SDPA RangeId 为 123、125、…、361；FFN 为 124、126、…、362。正式测量中的长尾全部保留。',
        '', '最新手写采集存在额外 GPU 延迟，SDPA 和 FFN 的均值均被长尾抬高；不能把均值比直接归因于 SDPA 实现差异。',
        '', '文件：`measured_calls.csv`（正式调用）、`excluded_warmup.csv`（排除的预热）、`summary.csv` / `summary.json`（统计及匹配校验）。','']
(OUT/'step2.md').write_text('\n'.join(lines))
print(json.dumps(result,ensure_ascii=False,indent=2))
