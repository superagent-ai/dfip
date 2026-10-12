"""Analyze exploratory adaptive results separately, preserving profile history."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('strengthened_analysis', ROOT/'scripts/analyze_strengthened.py')
analysis = importlib.util.module_from_spec(spec); spec.loader.exec_module(analysis)


def profile_record(record, profile):
    return {'model': record['model'], 'modes': {
        mode: {'phases': [p for p in report['phases'] if p['profile'] == profile]}
        for mode, report in record['modes'].items()}}


def validate_budget(report):
    finish = report.get('finish')
    if not finish:
        return
    for graph in finish['diagnostics']['graphs']:
        admission = graph['admission']
        if (not admission or graph['captures'] != graph['resident_entries']
                or not 0 <= graph['captures'] <= admission['capture_attempts'] <= admission['max_capture_attempts']):
            raise ValueError('Adaptive capture budget or no-eviction invariant violated')


def validate_trial(record):
    if record['status'] != 'complete':
        raise ValueError('Incomplete trial: ' + record.get('failure', record['status']))
    analysis.serving_rows([record])
    containers = set()
    for mode, report in record['modes'].items():
        expected = report['setup']['environment']['container_id']
        containers.add(expected)
        for phase in report['phases']:
            for row in phase['samples']:
                if row['status'] != 'ok':
                    continue
                if row['container_id'] != expected or row.get('worker_mode', mode) != mode:
                    raise ValueError('Worker identity or configured mode changed during traffic')
        finish = report['finish']
        if finish['mode'] != mode or finish.get('container_id', expected) != expected:
            raise ValueError('Worker identity or mode changed before final audit')
    if len(containers) != 1:
        raise ValueError('Modes used different worker containers')
    validate_budget(record['modes']['adaptive'])


def main():
    folder = ROOT/'benchmarks/results/adaptive'
    files = sorted(folder.glob('*-round-*.json'))
    records = [json.loads(p.read_text()) for p in files]
    rows = []; fidelity = []; workers = []; excluded = []
    for path, r in zip(files, records):
        # First check accounting and worker identity across the entire round.
        try:
            validate_trial(r)
        except ValueError as exc:
            excluded.append({'file': str(path.relative_to(ROOT)), 'model': r['model'],
                             'round': r['round'], 'raw_status': r['status'],
                             'validated_status': 'excluded', 'reason': str(exc)})
            continue
        for profile in ('diverse', 'mixed'):
            selected = profile_record(r, profile)
            rows.extend(dict(profile=profile, **row) for row in analysis.serving_rows([selected]))
            if {'native', 'adaptive'} <= selected['modes'].keys():
                selected['modes']['production'] = selected['modes'].pop('adaptive')
                compared = analysis.continuous_fidelity(selected)
                compared['tested_mode'] = 'adaptive'
                fidelity.append({'model': r['model'], 'round': r['round'],
                                 'profile': profile, 'comparison': compared})
        workers.append({'model': r['model'], 'round': r['round'], 'status': r['status'],
                        'mode_order': r['mode_order'], 'failure': r.get('failure'),
                        'modes': {mode: {'setup': report['setup'], 'finish': report.get('finish')}
                                  for mode, report in r['modes'].items()}})
    summary = {'scope': 'exploratory one-container pilots; never pooled with v3',
               'serving': rows, 'matched_fidelity': fidelity, 'workers': workers,
               'excluded_trials': excluded,
               'legacy_identity_scope': 'round 0 did not emit per-response modes; final state and container IDs still checked'}
    (ROOT/'paper/generated/adaptive-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    body = []
    for r in rows:
        if r['rate'] != 5:
            continue
        loss = 100*(r['dropped']+r['failed'])/r['offered']
        tail = f"{r['p95_ms_geomean']:.0f}" if r['p95_ms_geomean'] is not None else '--'
        body.append(f"{analysis.original.LABELS[r['model']]} & {r['profile']} & {r['mode']} & "
                    f"{r['mean_window_throughput_rps']:.2f} & {tail} & {loss:.1f} "+chr(92)*2)
    analysis.tabular(ROOT/'paper/generated/adaptive_rows.tex', 'lllrrr',
                     'Checkpoint & Trace & Mode & Completed/s & p95 (ms) & Loss (\\%)', body)
    evidence = sorted(p for p in folder.iterdir() if p.suffix in ('.json', '.jsonl') and p.name != 'sha256.json')
    (folder/'sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in evidence}, indent=2)+'\n')
    print(json.dumps({'records': len(records), 'rows': len(rows), 'statuses': [r['status'] for r in records]}))


if __name__ == '__main__':
    main()
