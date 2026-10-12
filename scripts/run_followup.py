import json
import subprocess
from pathlib import Path
from dfip.cloud.research import app, lifetime_probe, julia
from dfip.cloud.fast_baseline import app as fast_app, small
commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
folder=Path('benchmarks/results/follow-up');folder.mkdir(parents=True,exist_ok=True)
with app.run():
    result=lifetime_probe.remote()
    result['source_commit']=commit
    (folder/'kernel-retirement-validation.json').write_text(json.dumps(result,indent=2))
    print('Replay/kernel stress validation completed',flush=True)
    try:
        result=julia.remote('julia',0,50,True,commit)
        (folder/'julia-round-0.json').write_text(json.dumps(result,indent=2))
        print('Julia synchronized follow-up completed',flush=True)
    except Exception as exc:
        (folder/'julia-failure.txt').write_text(str(exc))
        print('Julia synchronized follow-up failed; retained diagnostic',flush=True)
with fast_app.run():
    try:
        result=small.remote('clef',0,50,True,commit)
        (folder/'clef-fast-round-0.json').write_text(json.dumps(result,indent=2))
        print('Clef optimized-convolution exploratory comparison completed',flush=True)
    except Exception as exc:
        (folder/'clef-fast-failure.txt').write_text(str(exc))
        print('Clef optimized-convolution exploratory comparison failed; retained diagnostic',flush=True)
