"""Serialize the two exploratory adaptive jobs and preserve exact source."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from run_strengthened import wait_retired

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume', required=True)
    args = parser.parse_args()
    modal = str(ROOT / '.venv/bin/modal')
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), DFIP_VOLUME=args.volume)
    wait_retired('dfip-study3', modal, env)
    folder = ROOT / 'benchmarks/results/adaptive'; folder.mkdir(parents=True, exist_ok=True)
    for model in ('decider', 'clef'):
        source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'src'], cwd=ROOT, check=True)
        archive = ROOT / 'benchmarks' / f'source-v4-{source[:12]}.zip'
        if not archive.exists():
            subprocess.run(['git', 'archive', '--format=zip', '--prefix=dfip-source/',
                            '--output='+str(archive), source, 'src', 'scripts', 'tests',
                            'pyproject.toml', 'benchmarks/adaptive-protocol.json'], cwd=ROOT, check=True)
        prefix = 'dfip-study4-' + model
        log = ROOT / 'work' / f'adaptive-{model}.log'
        started = time.time()
        print(json.dumps({'model': model, 'state': 'started', 'source_commit': source}), flush=True)
        with log.open('w') as stream:
            run = subprocess.run([modal, 'run', '-m', 'dfip.cloud.adaptive', '--model', model,
                                  '--commit', source], env=dict(env, DFIP_APP=prefix), cwd=ROOT,
                                 stdout=stream, stderr=subprocess.STDOUT)
        entry = {'model': model, 'source_commit': source, 'started_unix': started,
                 'finished_unix': time.time(), 'exit_code': run.returncode,
                 'source_archive': str(archive.relative_to(ROOT)), 'log': str(log.relative_to(ROOT))}
        with (folder / 'job-journal.jsonl').open('a') as stream:
            stream.write(json.dumps(entry)+'\n')
        print(json.dumps(entry), flush=True)
        wait_retired(prefix, modal, env)


if __name__ == '__main__':
    main()
