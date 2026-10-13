"""Reproduce the report from official Nsight CSV exports and SQLite traces.

Requires only Python's standard library. Original capture files are read-only.
Usage: python3 analyze_profiles.py --sqlite-dir /path/to/sqlite/files
"""
import argparse
import collections
import csv
import hashlib
import json
import re
import sqlite3
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'evidence'
LABELS = {'torch': 'Torch SDPA', 'my_latest': '手写 SDPA · 最新采集', 'my_earlier': '手写 SDPA · 较早采集'}
SQL_KERNELS = '''SELECT k.start, k.end, k.correlationId, k.gridX, k.gridY, k.gridZ,
 k.blockX, k.blockY, k.blockZ, k.streamId, s.value AS name
 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
 JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId
 AND (r.globalTid >> 24)=(k.globalPid >> 24)
 WHERE r.start>=? AND r.start<? AND (r.globalTid & 16777215)=?
 ORDER BY k.start'''

def read_csv(p):
    with p.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(name, rows):
    if not rows:
        return
    with (ROOT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def stats(values):
    return {'mean': statistics.mean(values), 'median': statistics.median(values),
            'std_population': statistics.pstdev(values), 'min': min(values), 'max': max(values)}

def kernel_category(name):
    if 'fmha_cutlass' in name:
        return '融合注意力'
    if 'gemm' in name.lower() or 'sgemm' in name.lower():
        return '矩阵乘法'
    if 'softmax' in name.lower():
        return 'softmax'
    if 'masked_fill' in name:
        return '因果掩码填充'
    if 'tril' in name.lower():
        return '构造下三角掩码'
    if 'FillFunctor' in name:
        return '张量填充'
    if 'MulFunctor' in name:
        return '逐元素乘法'
    if 'copy' in name.lower():
        return '张量拷贝'
    return '其他'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sqlite-dir', type=Path, default=ROOT / 'sqlite')
    args = parser.parse_args()
    manifest = json.loads((EVIDENCE / 'manifest.json').read_text())
    summaries, ranges, kernel_rows, groups, steps, checks, captures = [], [], [], [], [], {}, []
    signatures = {}
    for entry in manifest:
        capture = entry['name']
        db_path = args.sqlite_dir / (capture + '.sqlite')
        db = sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        logs = [dict(r) for r in db.execute('SELECT a.filename,s.value FROM ANALYSIS_FILE a LEFT JOIN StringIds s ON s.id=a.contentId')]
        stdout = '\n'.join(r['value'] or '' for r in logs if 'stdout' in r['filename'])
        total_line = next(x for x in stdout.splitlines() if x.startswith('total'))
        total_values = [float(x) for x in total_line.split()[1:]]
        timestamp = dict(db.execute('SELECT * FROM TARGET_INFO_SESSION_START_TIME').fetchone())
        benchmark = entry.get('benchmark')
        captures.append({'capture': capture, 'label': LABELS[capture], 'sha256': entry['sha256'],
                         'report_bytes': entry['report_bytes'], 'capture_utc': timestamp['utcTime'],
                         'benchmark_mean_ms': total_values[0], 'benchmark_std_ms': total_values[1],
                         'benchmark_min_ms': total_values[2], 'benchmark_max_ms': total_values[3],
                         'config': benchmark['config'] if benchmark else None,
                         'environment': benchmark['environment'] if benchmark else None})
        raw = read_csv(EVIDENCE / (capture + '_nvtx_gpu_proj_trace.csv'))
        by_module = collections.defaultdict(list)
        for r in raw:
            module = 'FFN' if r['Name'].endswith('ffn_forward') else 'SDPA'
            by_module[module].append(r)
        counts, mismatches = {}, []
        for module, source_rows in by_module.items():
            source_rows.sort(key=lambda r: int(r['Orig Start (ns)']))
            assert len(source_rows) == 180, (capture, module, len(source_rows))
            counts[module] = len(source_rows)
            measured = []
            for i, r in enumerate(source_rows):
                start = int(r['Orig Start (ns)']); end = start + int(r['Orig Duration (ns)'])
                ks = [dict(k) for k in db.execute(SQL_KERNELS, (start, end, int(r['TID'])))]
                ops = [(k['start'], k['end']) for k in ks]
                for table in ['CUPTI_ACTIVITY_KIND_MEMSET', 'CUPTI_ACTIVITY_KIND_MEMCPY']:
                    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                        ops += [(o['start'], o['end']) for o in db.execute(
                            f'''SELECT g.start,g.end FROM {table} g
                            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=g.correlationId
                            AND (r.globalTid >> 24)=(g.globalPid >> 24)
                            WHERE r.start>=? AND r.start<? AND (r.globalTid & 16777215)=?''',
                            (start, end, int(r['TID'])))]
                assert len(ops) == int(r['NumGPUOps']), (capture, module, i, len(ops), r['NumGPUOps'])
                reconstructed = max(o[1] for o in ops) - min(o[0] for o in ops)
                if reconstructed != int(r['Projected Duration (ns)']):
                    mismatches.append({'module':module,'call':i+1,'official_ns':int(r['Projected Duration (ns)']),'derived_ns':reconstructed})
                row = {'capture':capture,'label':LABELS[capture],'module':module,'range':r['Name'],
                       'call':i+1,'warmup':i<60,'step':i//12+1,'layer':i%12+1,
                       'original_start_ns':start,'projected_start_ns':int(r['Projected Start (ns)']),
                       'gpu_projected_us':int(r['Projected Duration (ns)'])/1000,
                       'cpu_range_us':int(r['Orig Duration (ns)'])/1000,
                       'kernel_sum_us':sum(k['end']-k['start'] for k in ks)/1000,
                       'gpu_ops':len(ops),'kernel_count':len(ks)}
                ranges.append(row)
                if i>=60:
                    measured.append(row)
                    for k in ks:
                        kernel_rows.append({'capture':capture,'label':LABELS[capture],'module':module,
                                            'call':i+1,'measured_step':(i-60)//12+1,'layer':i%12+1,
                                            'start_ns':k['start'],'end_ns':k['end'],'duration_us':(k['end']-k['start'])/1000,
                                            'name':k['name'],'category':kernel_category(k['name']),
                                            'grid':f"{k['gridX']}x{k['gridY']}x{k['gridZ']}",
                                            'block':f"{k['blockX']}x{k['blockY']}x{k['blockZ']}", 'stream':k['streamId']})
            summary={'capture':capture,'label':LABELS[capture],'module':module,'instances_all':180,'instances_measured':120,
                     'gpu_ops_per_call':statistics.mean(r['gpu_ops'] for r in measured),
                     'kernels_per_call':statistics.mean(r['kernel_count'] for r in measured)}
            for field in ['gpu_projected_us','cpu_range_us','kernel_sum_us']:
                for key,value in stats([r[field] for r in measured]).items():summary[field+'_'+key]=value
            per_step=[sum(r['gpu_projected_us'] for r in measured[j:j+12])/1000 for j in range(0,120,12)]
            summary['per_step_projected_ms_mean']=statistics.mean(per_step)
            summary['per_step_projected_ms_std_population']=statistics.pstdev(per_step)
            summaries.append(summary)
        checks[capture]={'nvtx_count':counts,'projection_reconstruction_mismatches':mismatches,
                         'report_stdout_success':not any(r['value'] for r in logs if 'stderr' in r['filename'])}
        sync=[dict(r) for r in db.execute("SELECT r.start,r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE 'cudaDeviceSynchronize%' ORDER BY r.start")]
        assert len(sync)>=16
        all_k=[dict(r) for r in db.execute('''SELECT k.start,k.end,s.value AS name,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ
                     FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
                     WHERE k.start>=? AND k.end<=? ORDER BY k.start''',(sync[-11]['end'],sync[-1]['end']))]
        signatures[capture]=[tuple(k[x] for x in ['name','gridX','gridY','gridZ','blockX','blockY','blockZ']) for k in all_k]
        for index,(previous,current) in enumerate(zip(sync[-11:-1],sync[-10:]),1):
            ks=[k for k in all_k if previous['end']<=k['start'] and k['end']<=current['end']]
            steps.append({'capture':capture,'label':LABELS[capture],'step':index,
                          'sync_interval_ms':(current['end']-previous['end'])/1e6,
                          'before_sync_ms':(current['start']-previous['end'])/1e6,
                          'sync_wait_ms':(current['end']-current['start'])/1e6,
                          'kernel_sum_ms':sum(k['end']-k['start'] for k in ks)/1e6,
                          'kernel_count':len(ks)})
        db.close()
    checks['same_handwritten_kernel_order_and_launch_shapes']=signatures['my_latest']==signatures['my_earlier']
    for (capture,module,name),rows in __import__('itertools').groupby(sorted(kernel_rows,key=lambda r:(r['capture'],r['module'],r['name'])),key=lambda r:(r['capture'],r['module'],r['name'])):
        values=list(rows); durations=[r['duration_us'] for r in values]
        out={'capture':capture,'label':LABELS[capture],'module':module,'name':name,'category':values[0]['category'],
             'instances':len(values),'total_ms':sum(durations)/1000,'sum_per_call_us':sum(durations)/120,
             'over_1ms':sum(x>1000 for x in durations)}
        out.update({key+'_us':value for key,value in stats(durations).items()})
        groups.append(out)
    result={'method':{'layers':12,'warmup_steps':5,'measured_steps':10,'drop_per_module':60,'keep_per_module':120,
                      'std_definition':'population, ddof=0','projection_definition':'official NVTX GPU Projection Trace; first GPU op start to last GPU op end',
                      'sync_interval_note':'Adjacent cudaDeviceSynchronize return timestamps include small loop overhead; diagnostic only.',
                      'sql_kernel_mapping':SQL_KERNELS},
            'captures':captures,'module_summary':summaries,'step_diagnostics':steps,'kernel_groups':groups,
            'nvtx_ranges':ranges,'measured_kernels':kernel_rows,'validation':checks}
    (ROOT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for name,rows in [('module_summary.csv',summaries),('nvtx_ranges.csv',ranges),('measured_kernels.csv',kernel_rows),('kernel_groups.csv',groups),('step_diagnostics.csv',steps)]:write_csv(name,rows)
    assert all(not c['projection_reconstruction_mismatches'] for k,c in checks.items() if isinstance(c,dict)), checks
    print(json.dumps({'module_summary':summaries,'validation':checks},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
