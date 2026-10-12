"""One-GPU exploratory comparison; kept separate from the frozen v3 study."""

import asyncio
import json
from pathlib import Path
import random
import time
import modal
from dfip.adaptive_protocol import PROTOCOL, mixed_record
from dfip.cloud.fast_baseline import fast_image
from dfip.cloud.images import volume
from dfip.cloud.sustained import phase
from dfip.config import settings
from dfip.study_protocol import request_record

app = modal.App(settings().app_name + '-adaptive-pilot')


def load(model):
    from dfip.runtime.adaptive_study import AdaptiveWorker
    volume.reload()
    return AdaptiveWorker(model)


@app.cls(image=fast_image, gpu='B200', cpu=8, memory=98304,
         volumes={'/research': volume}, max_containers=1, scaledown_window=300,
         timeout=180, startup_timeout=1200)
@modal.concurrent(max_inputs=64, target_inputs=64)
class Decider:
    @modal.enter()
    def start(self):
        self.worker = load('decider')

    @modal.method()
    def configure(self, mode):
        return self.worker.configure(mode)

    @modal.method()
    def predict(self, record):
        return self.worker.predict(record)

    @modal.method()
    def finish(self):
        return self.worker.finish()


@app.cls(image=fast_image, gpu='L40S', cpu=4, memory=32768,
         volumes={'/research': volume}, max_containers=1, scaledown_window=300,
         timeout=180, startup_timeout=1200)
@modal.concurrent(max_inputs=64, target_inputs=64)
class Clef:
    @modal.enter()
    def start(self):
        self.worker = load('clef')

    @modal.method()
    def configure(self, mode):
        return self.worker.configure(mode)

    @modal.method()
    def predict(self, record):
        return self.worker.predict(record)

    @modal.method()
    def finish(self):
        return self.worker.finish()


async def pilot(model, round_id, commit, duration):
    worker = {'decider': Decider, 'clef': Clef}[model]()
    folder = Path('benchmarks/results/adaptive'); folder.mkdir(parents=True, exist_ok=True)
    result = {'protocol': PROTOCOL, 'model': model, 'round': round_id,
              'source_commit': commit, 'status': 'running', 'modes': {},
              'duration_per_rate_seconds': duration,
              'scope': 'one exploratory container per model; no reliability certification',
              'client': {'transport': 'authenticated Modal RPC', 'max_in_flight': 128,
                         'server_max_concurrent_inputs': 64, 'gpu_execution': 'serialized'}}
    modes = ['native', 'adaptive']; random.Random(20261013 + round_id).shuffle(modes)
    result['mode_order'] = modes
    dest = folder / f'{model}-adaptive-round-{round_id}.json'

    def save():
        dest.write_text(json.dumps(result, indent=2))

    try:
        for mode in modes:
            start = time.perf_counter()
            report = {'setup': await worker.configure.remote.aio(mode), 'phases': []}
            report['client_setup_seconds'] = time.perf_counter() - start
            result['modes'][mode] = report; save()
            for profile_id, (profile, factory) in enumerate((('diverse', request_record),
                                                           ('mixed', mixed_record))):
                for phase_id, rate in enumerate((5, 15, 30)):
                    data = await phase(worker, rate, duration, profile_id * 3 + phase_id, factory,
                                       report['setup']['environment']['container_id'], mode)
                    # Actual worker state, rather than only the client's label, is checked.
                    data['profile'] = profile
                    report['phases'].append(data); save()
                    print(json.dumps({'model': model, 'mode': mode, 'profile': profile,
                                      'rate': rate, **data['summary']}), flush=True)
                    if data['summary']['failed']:
                        raise RuntimeError('Request error: stop and retain pilot failure')
            report['finish'] = await worker.finish.remote.aio(); save()
            if (report['finish']['container_id'] != report['setup']['environment']['container_id']
                    or report['finish']['mode'] != mode):
                raise RuntimeError('Worker identity or mode changed before final audit')
        result['status'] = 'complete'
    except Exception as exc:
        result['status'] = 'failed'; result['failure'] = str(exc)
    save()


@app.local_entrypoint()
def main(model: str = 'decider', round_id: int = 0, commit: str = 'unknown', duration: int = 60):
    asyncio.run(pilot(model, round_id, commit, duration))
