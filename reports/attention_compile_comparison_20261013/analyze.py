"""Analyze official NVTX projection CSVs and validate against CUDA launch correlations."""
import bisect
import collections
import csv
import hashlib
import json
import sqlite3
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT.parent
EVIDENCE = ROOT/'evidence' if (ROOT/'evidence').exists() else ROOT.parent
SQL = """SELECT k.start,k.end,k.correlationId,k.gridX,k.gridY,k.gridZ,
k.blockX,k.blockY,k.blockZ,k.streamId,s.value AS name,
r.start AS launch_start,r.globalTid
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON s.id=k.demangledName
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId
AND (r.globalTid >> 24)=(k.globalPid >> 24)
ORDER BY r.start,k.start"""

def stats(values):
    ordered = sorted(values)
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                std=statistics.pstdev(values), min=min(values), max=max(values),
                p95=ordered[round((len(ordered)-1)*.95)])

def write_csv(name, rows):
    with (ROOT/name).open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

result = dict(method=dict(warmup_steps=5, measured_steps=100, layers=12,
    exclude_first_ranges_per_module=60, measured_ranges_per_module=1200,
    projection_definition='first correlated GPU operation start to last operation end, including gaps',
    sql=SQL, measured_scope='attention forward only; full model forward/backward for benchmark total'), cases={})
all_ranges=[]
all_kernels=[]
all_groups=[]

for case in ('eager','compile'):
    prefix='nsys_attention_'+case+'_small_t512'
    benchmark=json.loads((OUTPUTS/(prefix+'.json')).read_text())
    assert benchmark['status']=='ok'
    db=sqlite3.connect((EVIDENCE/(prefix+'.sqlite')).as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    logs=[dict(row) for row in db.execute('SELECT a.filename,s.value FROM ANALYSIS_FILE a LEFT JOIN StringIds s ON s.id=a.contentId')]
    assert not any(row['value'] for row in logs if 'stderr' in row['filename'])
    stdout='\n'.join(row['value'] or '' for row in logs if 'stdout' in row['filename'])
    assert f"{benchmark['summary']['total']['mean_ms']:.3f}" in stdout
    kernels_by_tid=collections.defaultdict(list)
    for row in db.execute(SQL):
        row=dict(row)
        row['gpu_type']='kernel'
        kernels_by_tid[row['globalTid'] & 16777215].append(row)
    memop_sql=[]
    for table in ('CUPTI_ACTIVITY_KIND_MEMSET','CUPTI_ACTIVITY_KIND_MEMCPY'):
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone():
            query=f'''SELECT g.start,g.end,r.start AS launch_start,r.globalTid
                FROM {table} g JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
                ON r.correlationId=g.correlationId
                AND (r.globalTid >> 24)=(g.globalPid >> 24) ORDER BY r.start'''
            memop_sql.append(query)
            for row in db.execute(query):
                row=dict(row);row['gpu_type']=table
                kernels_by_tid[row['globalTid'] & 16777215].append(row)
    for rows in kernels_by_tid.values():rows.sort(key=lambda row:row['launch_start'])
    launch_starts={tid:[row['launch_start'] for row in rows] for tid,rows in kernels_by_tid.items()}
    raw=list(csv.DictReader((EVIDENCE/(prefix+'_trace_nvtx_gpu_proj_trace.csv')).open()))
    modules={}
    for module, label in [('attention','my_sdpa_torch_forward'),('ffn','ffn_forward')]:
        source=sorted((row for row in raw if row['Name']==':'+label),key=lambda row:int(row['Orig Start (ns)']))
        assert len(source)==1260,(case,module,len(source))
        expected_count=db.execute('SELECT count(*) FROM NVTX_EVENTS WHERE text=?',(label,)).fetchone()[0]
        assert expected_count==len(source)
        measured=[]
        measured_kernels=[]
        for i,row in enumerate(source):
            start=int(row['Orig Start (ns)']);end=start+int(row['Orig Duration (ns)']);tid=int(row['TID'])
            lo=bisect.bisect_left(launch_starts[tid],start)
            hi=bisect.bisect_left(launch_starts[tid],end)
            ops=kernels_by_tid[tid][lo:hi]
            ks=[op for op in ops if op['gpu_type']=='kernel']
            assert len(ops)==int(row['NumGPUOps']),(case,module,i,len(ops),row['NumGPUOps'])
            proj=max(k['end'] for k in ops)-min(k['start'] for k in ops)
            assert proj==int(row['Projected Duration (ns)'])
            assert min(k['start'] for k in ops)==int(row['Projected Start (ns)'])
            ksum=sum(k['end']-k['start'] for k in ks)
            out=dict(case=case,module=module,call=i+1,warmup=i<60,
                measured_step=(i-60)//12+1 if i>=60 else None,layer=i%12+1,
                orig_start_ns=start,cpu_us=int(row['Orig Duration (ns)'])/1000,
                projection_us=proj/1000,kernel_sum_us=ksum/1000,
                gap_us=(proj-sum(op['end']-op['start'] for op in ops))/1000,
                gpu_op_sum_us=sum(op['end']-op['start'] for op in ops)/1000,
                gpu_op_count=len(ops),kernel_count=len(ks))
            assert out['gap_us']>=0
            all_ranges.append(out)
            if i>=60:
                measured.append(out)
                for order,k in enumerate(ks):
                    ko=dict(case=case,module=module,measured_step=out['measured_step'],
                        layer=out['layer'],call=i+1,order=order+1,name=k['name'],
                        duration_us=(k['end']-k['start'])/1000,stream=k['streamId'],
                        grid=f"{k['gridX']}x{k['gridY']}x{k['gridZ']}",
                        block=f"{k['blockX']}x{k['blockY']}x{k['blockZ']}")
                    all_kernels.append(ko);measured_kernels.append(ko)
        assert len(measured)==1200
        metrics={field:stats([row[field] for row in measured]) for field in ('cpu_us','projection_us','kernel_sum_us','gpu_op_sum_us','gap_us','kernel_count','gpu_op_count')}
        perstep=[sum(row['projection_us'] for row in measured[i:i+12])/1000 for i in range(0,1200,12)]
        metrics['per_model_forward_attention_or_ffn_projection_ms']=stats(perstep)
        metrics['by_layer']={str(layer):stats([row['projection_us'] for row in measured if row['layer']==layer]) for layer in range(1,13)}
        metrics['total_projection_ms']=sum(row['projection_us'] for row in measured)/1000
        metrics['total_kernel_ms']=sum(row['kernel_sum_us'] for row in measured)/1000
        groups=collections.defaultdict(list)
        for row in measured_kernels:groups[row['name']].append(row['duration_us'])
        metrics['kernel_groups']=[]
        for name,values in sorted(groups.items(),key=lambda item:sum(item[1]),reverse=True):
            group=dict(case=case,module=module,name=name,count=len(values),total_ms=sum(values)/1000,
                       contribution_per_call_us=sum(values)/1200,**stats(values))
            metrics['kernel_groups'].append(group);all_groups.append(group)
        metrics['validation']=dict(total_ranges=1260,warmup_ranges=60,measured_ranges=1200,
            projection_reconstruction_mismatches=0,gpu_ops_equal_correlated_ops_count=True,
            memop_sql=memop_sql)
        modules[module]=metrics
    capture=dict(db.execute('SELECT * FROM TARGET_INFO_SESSION_START_TIME').fetchone())
    result['cases'][case]=dict(config=benchmark['config'],environment=benchmark['environment'],
        benchmark_total_ms=benchmark['summary']['total'],capture=capture,
        report_sha256=hashlib.sha256((OUTPUTS/(prefix+'.nsys-rep')).read_bytes()).hexdigest(),modules=modules)
    db.close()

a=result['cases']['eager'];b=result['cases']['compile']
skip={'output','if_compile_sdpa'}
assert {k:v for k,v in a['config'].items() if k not in skip}=={k:v for k,v in b['config'].items() if k not in skip}
assert a['environment']==b['environment']
result['comparison']={}
for field in ('projection_us','kernel_sum_us','gap_us','cpu_us'):
    av=a['modules']['attention'][field]['mean'];bv=b['modules']['attention'][field]['mean']
    result['comparison'][field]=dict(eager=av,compile=bv,speedup=av/bv,reduction_pct=(1-bv/av)*100)
av=a['benchmark_total_ms']['mean_ms'];bv=b['benchmark_total_ms']['mean_ms']
result['comparison']['model_total_ms']=dict(eager=av,compile=bv,speedup=av/bv,reduction_pct=(1-bv/av)*100)
(ROOT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
write_csv('nvtx_ranges.csv',all_ranges)
write_csv('measured_kernels.csv',all_kernels)
write_csv('kernel_groups.csv',all_groups)
print(json.dumps(dict(comparison=result['comparison'], attention={case:result['cases'][case]['modules']['attention'] for case in result['cases']}),ensure_ascii=False,indent=2))
