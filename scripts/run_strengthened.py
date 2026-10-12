"""Replay the new study serially, preserving failed jobs and enforcing GPU retirement."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'benchmarks/results/strengthened'


def wait_retired(app_name, executable, env):
    deadline=time.monotonic()+180
    while True:
        process=subprocess.run([executable,'app','list','--json'],env=env,cwd=ROOT,
                               capture_output=True,text=True,check=True)
        apps=json.loads(process.stdout)
        active=[a for a in apps if a['description'].startswith(app_name)
                and int(a.get('tasks') or 0)>0]
        if not active:
            return
        if time.monotonic()>deadline:
            raise RuntimeError('Prior study application still has active workers; refusing overlapping GPUs')
        time.sleep(5)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--modal',default=str(ROOT/'.venv/bin/modal'))
    parser.add_argument('--volume',required=True)
    parser.add_argument('--wait-for-pid',type=int)
    parser.add_argument('--wait-app-prefix',default='dfip-inference-paper-strengthened')
    parser.add_argument('--skip-decider-baseline',action='store_true')
    parser.add_argument('--skip-baselines',action='store_true')
    args=parser.parse_args()
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),DFIP_VOLUME=args.volume)
    FOLDER.mkdir(parents=True,exist_ok=True)
    if args.wait_for_pid:
        while True:
            try:
                os.kill(args.wait_for_pid,0)
            except ProcessLookupError:
                break
            time.sleep(5)
        wait_retired(args.wait_app_prefix,args.modal,env)
    jobs=[]
    models=() if args.skip_baselines else (('clef',) if args.skip_decider_baseline else ('decider','clef'))
    for model in models:
        jobs.append((f'{model}-optimized','dfip.cloud.strengthen',
                     ['--action','baseline','--model',model,'--rounds','5']))
    for model in ('decider','clef'):
        jobs.append((f'{model}-readout-audit','dfip.cloud.strengthen',
                     ['--action','audit','--model',model,'--rounds','1']))
    jobs.append(('liquid-compiler','dfip.cloud.strengthen',
                 ['--action','compiler','--model','liquid','--rounds','3']))
    for model in ('decider','clef','security'):
        for round_id in range(3):
            jobs.append((f'{model}-serving-{round_id}','dfip.cloud.sustained',
                         ['--model',model,'--round-id',str(round_id),'--duration','60']))
    journal=FOLDER/'job-journal.jsonl'
    for name,module,options in jobs:
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        subprocess.run(['git','diff','--quiet','HEAD','--','src'],cwd=ROOT,check=True)
        archive=ROOT/'benchmarks'/f'source-v3-{source[:12]}.zip'
        if not archive.exists():
            subprocess.run(['git','archive','--format=zip','--prefix=dfip-source/',
                            '--output='+str(archive),source,'src','scripts/run_strengthened.py',
                            'scripts/analyze_strengthened.py','benchmarks/strengthened-protocol.json',
                            'pyproject.toml'],cwd=ROOT,check=True)
        app_name='dfip-study3-'+name
        job_env=dict(env,DFIP_APP=app_name)
        log=ROOT/'work'/f'strengthened-{name}.log'
        started=time.time()
        print(json.dumps({'job':name,'state':'started','source_commit':source}),flush=True)
        with log.open('w') as output:
            process=subprocess.run([args.modal,'run','-m',module,*options,'--commit',source],
                                   env=job_env,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        record={'job':name,'exit_code':process.returncode,'source_commit':source,
                'started_unix':started,'finished_unix':time.time(),
                'log':str(log.relative_to(ROOT)),'app_prefix':app_name}
        record['source_archive']=str(archive.relative_to(ROOT))
        with journal.open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        wait_retired(app_name,args.modal,job_env)


if __name__=='__main__':
    main()
