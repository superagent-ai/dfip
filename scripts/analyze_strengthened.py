"""Generate strengthened-study tables from raw records, without Torch or GPUs."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('original_analysis',ROOT/'scripts/analyze_paper.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
from dfip.study_protocol import summarize_phase, compare_reported_answers


def median(row):
    samples=row['samples_ms']
    if len(samples)!=50 or any(not math.isfinite(x) or x<=0 for x in samples):
        raise ValueError('Invalid optimized-native samples')
    measured=statistics.median(samples)
    if not math.isclose(measured,row['p50_ms'],rel_tol=1e-10):
        raise ValueError('Inconsistent optimized-native median')
    return measured


def paired(records,reference,tested):
    results=[]
    common=set.intersection(*(set(t['scenario'] for t in r['modes'][reference]['timings'])
                              for r in records))
    for scenario in original.SCENARIOS:
        if scenario not in common:
            continue
        base=[median(next(t for t in r['modes'][reference]['timings'] if t['scenario']==scenario)) for r in records]
        opt=[median(next(t for t in r['modes'][tested]['timings'] if t['scenario']==scenario)) for r in records]
        ratios=[a/b for a,b in zip(base,opt)]
        speed,lo,hi=original.interval(ratios)
        results.append({'scenario':scenario,'rounds':len(records),
                        'reference_mode':reference,'tested_mode':tested,
                        'native_ms':original.geometric_mean(base),
                        'tested_ms':original.geometric_mean(opt),
                        'speedup':speed,'ci_lower':lo if len(records)>1 else None,
                        'ci_upper':hi if len(records)>1 else None,'round_ratios':ratios})
    return results


def serving_rows(records):
    groups={}
    for record in records:
        containers=set()
        for mode,report in record['modes'].items():
            for phase in report['phases']:
                summary=phase['summary']
                computed=summarize_phase(phase['samples'],summary['duration_seconds'],summary['offered_rps'])
                if computed!=summary:
                    raise ValueError('Serving summary differs from raw requests')
                containers.update(r['container_id'] for r in phase['samples'] if r['status']=='ok')
                key=(record['model'],mode,summary['offered_rps'])
                groups.setdefault(key,[]).append(summary)
        if len(containers)>1:
            raise ValueError('A serving round used more than one worker container')
    rows=[]
    for (model,mode,rate),summaries in sorted(groups.items()):
        p95=[s['p95_ms'] for s in summaries if s['p95_ms'] is not None]
        p99=[s['p99_ms'] for s in summaries if s['p99_ms'] is not None]
        rows.append({'model':model,'mode':mode,'rate':rate,'rounds':len(summaries),
                     'offered':sum(s['offered'] for s in summaries),
                     'successful':sum(s['successful'] for s in summaries),
                     'dropped':sum(s['dropped'] for s in summaries),
                     'failed':sum(s['failed'] for s in summaries),
                     'p95_ms_geomean':original.geometric_mean(p95) if p95 else None,
                     'p95_ms_range':[min(p95),max(p95)] if p95 else None,
                     'p99_ms_geomean':original.geometric_mean(p99) if p99 else None,
                     'mean_window_throughput_rps':statistics.mean(s['throughput_within_window_rps'] for s in summaries),
                     'max_drain_seconds':max(s['drain_seconds'] for s in summaries),
                     'max_dispatch_lag_ms':max(s['max_dispatch_lag_ms'] for s in summaries),
                     'raw_round_summaries':summaries})
    return rows


def continuous_fidelity(record):
    """Match request IDs within each rate phase; retain missing/drop counts separately."""
    tested_mode='production' if 'production' in record['modes'] else 'native_concurrent'
    if not {'native',tested_mode}<=record['modes'].keys():
        return None
    reference={}
    tested={}
    for mode,target in [('native',reference),(tested_mode,tested)]:
        for phase in record['modes'][mode]['phases']:
            for request in phase['samples']:
                if request['status']=='ok':
                    for p in request['prediction']:
                        if p['id'] in target:
                            raise ValueError('Duplicate serving decision ID')
                        target[p['id']]=p
    shared=reference.keys() & tested.keys()
    drift=[]; flips=[]; crossings={str(t):0 for t in (.1,.25,.5,.75,.9)}
    for identity in shared:
        a,b=reference[identity],tested[identity]
        if a['probabilities'].keys()!=b['probabilities'].keys():
            raise ValueError('Serving candidate schema changed')
        if a['prediction']!=b['prediction']:
            flips.append(identity)
        for key,p in a['probabilities'].items():
            q=b['probabilities'][key];drift.append(abs(p-q))
            for threshold in crossings:
                crossings[threshold]+=int((p>=float(threshold))!=(q>=float(threshold)))
    return {'tested_mode':tested_mode,'matched_decisions':len(shared),'native_unmatched':len(reference)-len(shared),
            'tested_unmatched':len(tested)-len(shared),'argmax_flips':len(flips),
            'flip_ids':sorted(flips),'max_probability_drift':max(drift,default=0),
            'threshold_crossings':crossings,
            'reported_api':compare_reported_answers([reference[k] for k in sorted(shared)],
                                                    [tested[k] for k in sorted(shared)])}


def tabular(path,columns,header,body):
    path.write_text('\\begin{tabular}{'+columns+'}\n\\toprule\n'+header+' '+chr(92)*2+
                    '\n\\midrule\n'+'\n'.join(body)+'\n\\bottomrule\n\\end{tabular}\n')


def main():
    folder=ROOT/'benchmarks/results/strengthened'
    generated=ROOT/'paper/generated'
    files=sorted(folder.glob('*-round-*.json'))
    records=[json.loads(p.read_text()) for p in files]
    optimized=[]; compiler=[]; failures=[]; serving=[]
    for r in records:
        if r['status']=='failed':
            failures.append({'model':r.get('model',r.get('model_id')),'round':r['round'],
                             'protocol':r['protocol'],'failure':r.get('failure','unknown'),
                             'retained_modes':list(r['modes'])})
        if r['protocol']=='dfip-paper-v3-external-serving':
            serving.append(r)
    for model in ('clef','decider'):
        valid=[r for r in records if r['protocol']=='dfip-paper-v3-optimized-native'
               and r['model']==model and r['status']=='complete']
        if valid:
            optimized.extend(dict(model=model,**r) for r in paired(valid,'native','production'))
    liquid=[r for r in records if r['protocol']=='dfip-paper-v3-released-compiler'
            and r['status']=='complete']
    if liquid:
        compiler=paired(liquid,'native_compile','production')
    serving_summary=serving_rows(serving)
    summary={'optimized_native':optimized,'released_compiler':compiler,
             'literal_readout_audits':[{'model':r['model'],'status':r['status'],
                                        'comparison':r.get('comparison')}
                                       for r in records if r['protocol']=='dfip-paper-v3-literal-readout-audit'],
             'serving':serving_summary,'failures':failures,
             'serving_fidelity':[{'model':r['model'],'round':r['round'],
                                  'comparison':continuous_fidelity(r)} for r in serving],
             'status_counts':{m:{'complete':sum(r['status']=='complete' for r in records if r.get('model')==m),
                                 'failed':sum(r['status']=='failed' for r in records if r.get('model')==m)}
                              for m in ('clef','decider','liquid','security')}}
    (generated/'strengthened-summary.json').write_text(json.dumps(summary,indent=2))
    body=[]
    for r in optimized:
        if r['scenario']=='short_binary':
            ci=f"[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]" if r['ci_lower'] is not None else 'single round'
            body.append(f"{original.LABELS[r['model']]} & {r['rounds']}/5 & {r['native_ms']:.2f} & {r['tested_ms']:.2f} & {r['speedup']:.2f} {ci} "+chr(92)*2)
    tabular(generated/'optimized_native_rows.tex','lrrrr',
            'Checkpoint & Rounds & Native (ms) & DFIP (ms) & Speedup [95\\% CI]',body)
    body=[]
    for r in serving_summary:
        loss=100*(r['dropped']+r['failed'])/r['offered']
        p95=f"{r['p95_ms_geomean']:.0f}" if r['p95_ms_geomean'] is not None else '--'
        mode={'production':'graph','native':'native','native_concurrent':'parallel'}[r['mode']]
        body.append(f"{original.LABELS[r['model']]} & {mode} & {r['rate']} & {r['rounds']} & {r['mean_window_throughput_rps']:.1f} & {p95} & {loss:.1f} "+chr(92)*2)
    tabular(generated/'serving_rows.tex','llrrrrr',
            'Checkpoint & Mode & Offered/s & $n$ & Completed/s & p95 (ms) & Loss (\\%)',body)
    (folder/'sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    print(json.dumps({'raw_records':len(records),'optimized_rows':len(optimized),
                      'serving_rows':len(serving_summary),'failures':len(failures)}))


if __name__=='__main__':
    main()
