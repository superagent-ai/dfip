"""Recompute pilot summaries from every archived attempt; never pool with v3/v4."""
import hashlib
import json
from pathlib import Path
import statistics

from dfip.contract_protocol import compare

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'benchmarks/results/contracts'
NAMES = {'omni': 'd1-omni-600M', 'gliner': 'GLiNER2.5-Decide', 'd3': 'd3-flash', 'jev9': 'JEV-9B', 'jev27': 'JEV-27B'}


def combine(reports):
    reports = list(reports)
    if not reports:
        return None
    result = {}
    for key in reports[0]:
        if key == 'passes':
            result[key] = all(r[key] for r in reports)
        elif key.startswith('max_'):
            result[key] = max(r[key] for r in reports)
        else:
            result[key] = sum(r[key] for r in reports)
    result['comparisons'] = len(reports)
    result['failed_comparisons'] = sum(not r['passes'] for r in reports)
    return result


def serving_summary(report, reference=None):
    completed = report['completed']; dropped = report['drops']
    indices = [r['index'] for r in completed] + [r['index'] for r in dropped]
    if sorted(indices) != list(range(report['offered'])) or report['completed_after_drain'] != len(completed):
        raise ValueError('Serving requests missing or duplicated')
    if report['completed_within_window'] != sum(r['finish_s'] <= report['arrival_horizon_s'] for r in completed):
        raise ValueError('Within-window completion count inconsistent')
    if sorted(i for b in report['batches'] for i in b['indices']) != sorted(r['index'] for r in completed):
        raise ValueError('Batch accounting mismatch')
    keys = ['offered', 'arrival_horizon_s', 'completed_within_window', 'completed_after_drain', 'elapsed_s',
            'errors', 'max_queue_depth', 'p50_ms', 'p95_ms', 'p99_ms', 'batched_fraction']
    comparisons = []
    for r in completed:
        comparison = compare(reference[r['index']], r['output']) if reference is not None else r['comparison']
        if comparison != r['comparison']:
            raise ValueError('Archived serving comparison inconsistent with reference')
        comparisons.append(comparison)
    return {**{k: report[k] for k in keys}, 'drops': len(dropped),
            'fidelity': combine(comparisons),
            'reference_recomputed': reference is not None}


def main():
    attempts = []; native = []; shared = []
    for path in sorted(FOLDER.glob('*.json')):
        record = json.loads(path.read_text())
        if record.get('action') not in ('native', 'shared', 'shared_compiled'):
            continue
        identity = {'file': str(path.relative_to(ROOT)), 'alias': record['alias'], 'round': record['round'], 'action': record['action'],
                    'source_commit': record['source_commit'], 'status': record['status'],
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'error': record.get('error')}
        attempts.append(identity)
        if record['status'] != 'complete':
            continue
        if record['action'] == 'native':
            native.append({**identity, 'load_seconds': record['load_seconds'], 'environment': record['environment'],
                           'repeatability': combine(compare(c['reference'], c['repeat']) for c in record['conformance']),
                           'resident_median_ms': {k: v['median_ms'] for k, v in record['resident'].items()},
                           'batch_probes': [{'size': b['size'], 'speedup_single_probe': b['speedup'],
                                             'fidelity': compare(b['reference'], b['outputs'])} for b in record['batch_probes']],
                           'profile': record['profile'], 'native_runtime': record.get('native_runtime')})
        else:
            resident = {}
            for name, cell in record['resident_shared'].items():
                medians = {mode: statistics.median(samples) for mode, samples in cell['samples_ms'].items()}
                if medians != cell['median_ms']:
                    raise ValueError('Resident median inconsistent')
                resident[name] = {'median_ms': medians, 'serial_to_shared': medians['serial']/medians['shared'],
                                  'native_batch_to_shared': medians['native_batch']/medians['shared'],
                                  'fidelity': {mode: compare(cell['reference'], audit['outputs'])
                                               for mode, audit in cell['audits'].items()},
                                  'shared_batch_fraction': sum(len(b) for b in cell['audits']['shared']['batches'] if len(b)>1)/len(cell['records'])}
            shared.append({**identity, 'load_seconds': record['load_seconds'], 'environment': record['environment'],
                           'admission_seconds': record['admission']['seconds'], 'admitted_keys': record['admission']['keys'],
                           'passed_size_gates': sum(e['admitted'] for e in record['admission']['evidence']),
                           'attempted_size_gates': len(record['admission']['evidence']),
                           'resident': resident,
                           'serving': {name: {mode: serving_summary(report, trace.get('reference')) for mode, report in trace['modes'].items()}
                                       for name, trace in record['serving_shared'].items()}, 'profile': record.get('profile'),
                           'compiler': record.get('compiler')})
    summary = {'scope': 'v5 exploratory; one native and one shared worker per checkpoint except retained retries; no independent speedup interval or production claim',
               'attempts': attempts, 'native': native, 'shared': shared,
               'export_failure': 'D3 native round 1 completed on GPU and was recovered from committed Volume after local Torch pickle dependency failure; batch adapter used wrong input-token field, retained diagnostically and repeated in round 2.'}
    generated = ROOT / 'paper/generated'; generated.mkdir(exist_ok=True)
    (generated/'contracts-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    # Latest corrected native attempt per checkpoint. Earlier failures and
    # superseded diagnostics remain in the summary and raw namespace.
    latest = {r['alias']: r for r in sorted(native, key=lambda r: r['round'])}
    rows = ['\\begin{tabular}{lrrrr}', '\\toprule', 'Checkpoint & Repeat fails & Native (ms) & Batch $8\\times$ & Drift \\\\', '\\midrule']
    for alias in NAMES:
        if alias not in latest:
            continue
        r = latest[alias]; batch = r['batch_probes'][-1]
        rows.append(f"{NAMES[alias]} & {r['repeatability']['failed_comparisons']}/29 & {r['resident_median_ms']['choice_short']:.2f} & {batch['speedup_single_probe']:.2f} & {batch['fidelity']['max_numeric_drift']:.3g} \\\\")
    rows += ['\\bottomrule', '\\end{tabular}']
    (generated/'contracts_native_rows.tex').write_text('\n'.join(rows)+'\n')
    rows = ['\\begin{tabular}{llrrrr}', '\\toprule', 'Checkpoint & Trace & Gates & Serial/shared & Batch/shared & Shared audit \\\\', '\\midrule']
    shared.sort(key=lambda r: (r['action'] != 'shared', list(NAMES).index(r['alias']), r['round']))
    for r in shared:
        label = NAMES[r['alias']] + (' (compiled)' if r['action'] == 'shared_compiled' else '')
        for name, cell in r['resident'].items():
            verdict = 'pass' if cell['fidelity']['shared']['passes'] else 'fail'
            rows.append(f"{label} & {name} & {r['passed_size_gates']}/{r['attempted_size_gates']} & {cell['serial_to_shared']:.2f} & {cell['native_batch_to_shared']:.2f} & {verdict} \\\\")
    rows += ['\\bottomrule', '\\end{tabular}']
    (generated/'contracts_shared_rows.tex').write_text('\n'.join(rows)+'\n')
    rows = ['\\begin{tabular}{lrrrrrr}', '\\toprule',
            'Checkpoint & \\multicolumn{3}{c}{Diverse p95 (ms)} & \\multicolumn{3}{c}{Drops / offered} \\\\',
            '& Serial & Batch & Shared & Serial & Batch & Shared \\\\', '\\midrule']
    for r in shared:
        label = NAMES[r['alias']] + (' (compiled)' if r['action'] == 'shared_compiled' else '')
        modes = r['serving']['diverse']; order = ['serial', 'native_batch', 'shared']
        latency = ' & '.join(f"{modes[m]['p95_ms']:.1f}" for m in order)
        drops = ' & '.join(f"{modes[m]['drops']}/{modes[m]['offered']}" for m in order)
        rows.append(f"{label} & {latency} & {drops} \\\\")
    rows += ['\\bottomrule', '\\end{tabular}']
    (generated/'contracts_serving_rows.tex').write_text('\n'.join(rows)+'\n')
    primary = {r['alias']: r for r in shared if r['action'] == 'shared'}
    findings = []
    if set(primary) == set(NAMES):
        g = primary['gliner']; h = g['resident']['homogeneous']; d = g['resident']['diverse']
        traffic = g['serving']['diverse']
        findings.append(
            f"GLiNER passes all {g['passed_size_gates']} schema/size gates and the exposed-label/confidence audits. "
            f"Shared scheduling improves the resident median by {h['native_batch_to_shared']:.2f}$\\times$ on recurring schemas and "
            f"{d['native_batch_to_shared']:.2f}$\\times$ on diverse schemas relative to native batching "
            f"({h['serial_to_shared']:.2f}$\\times$ and {d['serial_to_shared']:.2f}$\\times$ relative to serial). "
            "This is an incremental scheduling result in one worker, not evidence of a new GPU kernel. "
            f"On diverse scheduled arrivals, p95 is {traffic['shared']['p95_ms']:.1f}\\,ms for shared scheduling and "
            f"{traffic['native_batch']['p95_ms']:.1f}\\,ms for native batching; both complete all 256 requests without drops. "
            "The resident gain therefore does not establish a tail-latency advantage over native batching in this replay.")
        findings.append(
            "Omni, D3 and both JEV sizes reject every tested shared batch size under the frozen gate and execute single-request native fallback. "
            "Omni and D3 preserve the held-out shared answer fields but gain no meaningful speed over serial, and are slower than ungated native batching. "
            "Both JEV native-serial and shared-fallback audits exceed numerical tolerances; JEV-27B also changes selected answers during serving. "
            "These findings do not isolate an optimizer-induced numerical error, since shared batching was never admitted. "
            "The strict gate is unsuitable as a universal equivalence certificate when the native reference itself varies.")
        findings.append(
            "Table~\\ref{tab:contracts-serving} reports completed-request latency together with all drops. "
            "The conservative fallback still overloads on the diverse trace: D3 drops 41/256 requests, JEV-9B 132/256 and JEV-27B 61/256. "
            "Ungated native batching completes all offered requests for these models, but fails the exposed-answer fidelity gate. "
            "Choosing a slower path preserves neither a latency objective nor complete request delivery by itself. "
            "A useful general optimizer must improve on the strongest native baseline under an application-specific answer contract, rather than count native fallback as an acceleration.")
    compiled = [r for r in shared if r['action'] == 'shared_compiled']
    for r in compiled:
        d = r['resident']['diverse']; t = r['serving']['diverse']
        verdict = 'passes' if all(c['fidelity']['shared']['passes'] for c in r['resident'].values()) else 'fails'
        findings.append(
            "A post hoc GLiNER follow-up calls the released FP32 \\texttt{Extractor.compile()} before repeating the same policy and fixtures. "
            "All three modes then use that compiled substrate, with eager references collected before compilation in the worker. "
            f"Shared scheduling admits {r['passed_size_gates']}/{r['attempted_size_gates']} gates and {verdict} the resident audit; "
            f"its diverse resident ratio is {d['native_batch_to_shared']:.2f}$\\times$ relative to compiled native batching. "
            f"Diverse p95 is {t['shared']['p95_ms']:.1f}\\,ms for shared and {t['native_batch']['p95_ms']:.1f}\\,ms for compiled native batching. "
            "The fixtures were reused after inspecting the primary result, so this is a stronger-baseline diagnostic rather than independent confirmation. "
            "Compiler invocation and first-use compilation in admission are retained outside the warm resident ratios.")
    (generated/'contracts_findings.tex').write_text('\n\n'.join(findings)+'\n')
    print(json.dumps({'native_complete': len(native), 'shared_complete': len(shared), 'failed': sum(r['status']!='complete' for r in attempts)}))


if __name__ == '__main__':
    main()
