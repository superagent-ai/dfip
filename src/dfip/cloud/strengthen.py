"""Serialized strengthened-paper experiments; no production deployments changed."""

import modal
from dfip.cloud.images import image, volume
from dfip.cloud.fast_baseline import fast_image
from dfip.config import settings

app = modal.App(settings().app_name + "-paper-strengthened")


def baseline_run(model, round_id, commit):
    import json
    import traceback
    from pathlib import Path
    from dfip.runtime.research import round_run

    volume.reload()
    folder = Path('/research/results/strengthened-v3')
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f'{model}-optimized-round-{round_id}.json'
    result = {'protocol':'dfip-paper-v3-optimized-native', 'model':model,
              'round':round_id, 'source_commit':commit, 'status':'running'}

    def progress(value):
        result.update(value)
        result['protocol'] = 'dfip-paper-v3-optimized-native'
        result['status'] = 'running'
        dest.write_text(json.dumps(result, indent=2))
        volume.commit()

    try:
        progress(round_run(model, round_id, 50, round_id == 0, commit, progress=progress))
        result['status'] = 'complete'
    except Exception:
        result['status'] = 'failed'
        result['failure'] = traceback.format_exc()
    dest.write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(image=fast_image, gpu='B200', cpu=8, memory=98304,
              volumes={'/research':volume}, timeout=7200, max_containers=1,
              single_use_containers=True)
def baseline_big(model, round_id, commit):
    return baseline_run(model, round_id, commit)


@app.function(image=fast_image, gpu='L40S', cpu=4, memory=32768,
              volumes={'/research':volume}, timeout=7200, max_containers=1,
              single_use_containers=True)
def baseline_small(model, round_id, commit):
    return baseline_run(model, round_id, commit)


@app.function(image=fast_image, gpu='L40S', cpu=4, memory=32768,
              volumes={'/research':volume}, timeout=3600, max_containers=1,
              single_use_containers=True)
def liquid_compiler(round_id, commit):
    import json
    import random
    import time
    import traceback
    from pathlib import Path
    import torch
    from dfip.runtime.research import AblationEngine, manifest, measure, compare, workloads
    from dfip.runtime.benchmark import distributions, sync
    from dfip.study_protocol import records_for_fidelity
    from dfip.models import get_model

    volume.reload(); torch.set_num_threads(4)
    result={'protocol':'dfip-paper-v3-released-compiler','source_commit':commit,
            'model':'liquid','round':round_id,'status':'running','modes':{},
            'environment':manifest(),'compiler':'torch.compile(mode=reduce-overhead), released runner setting'}
    folder=Path('/research/results/strengthened-v3');folder.mkdir(parents=True,exist_ok=True)
    dest=folder/f'liquid-compiler-round-{round_id}.json'
    def save():
        dest.write_text(json.dumps(result,indent=2));volume.commit()
    save()
    try:
        start=time.perf_counter()
        engine=AblationEngine(get_model('liquid').model_id);sync()
        result['load_seconds']=time.perf_counter()-start
        graph=engine.adapter.engine._one_pass
        compiled=None
        cases=[w for w in workloads() if w['name'] in ('short_binary','short_choice','short_3_questions')]
        records=records_for_fidelity()
        engine.configure('native')
        reference=[]
        for record in records:
            reference.extend(distributions([record],engine.predict([record])))
        modes=['native','native_compile','production']
        random.Random(20261012+round_id).shuffle(modes);result['mode_order']=modes;save()
        for mode in modes:
            result['active_mode']=mode;save()
            engine.adapter.engine._one_pass=graph
            engine.configure('native' if mode=='native_compile' else mode)
            if mode=='native_compile':
                if compiled is None:
                    compiled=torch.compile(engine.adapter.model.forward,mode='reduce-overhead')
                engine.adapter.engine._one_pass=compiled
                engine.mode=mode
            timings=[measure(engine,w,50) for w in cases]
            actual=[]
            for record in records:
                actual.extend(distributions([record],engine.predict([record])))
            result['modes'][mode]={'timings':timings,'fidelity':compare(reference,actual),
                                   'predictions':actual};save()
        result['status']='complete'
    except Exception:
        result['status']='failed';result['failure']=traceback.format_exc()
    save();return result


@app.function(image=image, volumes={"/research": volume}, timeout=300, max_containers=1)
def inventory():
    import json
    from pathlib import Path
    from dfip.models import MODELS
    result = {}
    for model in MODELS:
        meta = json.loads((Path('/research/prepared') / (model.model_id.replace('/', '__') + '.json')).read_text())
        root = Path(meta['path'])
        names = [str(p.relative_to(root)) for p in root.rglob('*.py')]
        chosen = []
        for name in names:
            if name.endswith(('engine.py','model.py','modeling_d1.py','joint_schema_model.py','runner.py','api.py')):
                source = (root/name).read_text()
                lines = source.splitlines()
                matches = [i for i,line in enumerate(lines) if any(k in line for k in ('def compile','def load_release_model','torch.compile','attn_implementation','marker_only_head','causal_conv1d','flash_attn'))]
                chosen.append({'file':name,'excerpts':['\n'.join(lines[max(0,i-3):i+12]) for i in matches][:15]})
        result[model.alias] = {'files':names,'optimizations':chosen}
    return result


@app.local_entrypoint()
def main(action: str = 'inventory', model: str = 'decider', rounds: int = 5,
         commit: str = 'unknown'):
    import json
    from pathlib import Path
    if action == 'inventory':
        result = inventory.remote()
        Path('work/native-inventory.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    if action not in ('baseline','compiler') or model not in ('clef', 'decider','liquid'):
        raise ValueError((action,model))
    folder=Path('benchmarks/results/strengthened'); folder.mkdir(parents=True,exist_ok=True)
    function = baseline_big if model == 'decider' else baseline_small
    for round_id in range(rounds):
        result = liquid_compiler.remote(round_id,commit) if action=='compiler' else function.remote(model, round_id, commit)
        dest=folder/(f'liquid-compiler-round-{round_id}.json' if action=='compiler' else f'{model}-optimized-round-{round_id}.json')
        dest.write_text(json.dumps(result,indent=2))
        print(json.dumps({'saved':str(dest),'status':result['status']}),flush=True)
