"""One-GPU, authenticated external-client sustained serving experiment."""

import asyncio
import json
import math
from pathlib import Path
import random
import time
import modal
from dfip.cloud.fast_baseline import fast_image
from dfip.cloud.images import sglang_image, volume
from dfip.config import settings
from dfip.study_protocol import SERVING_PROTOCOL, request_record, summarize_phase

app=modal.App(settings().app_name+'-paper-sustained')


def load_worker(model):
    from dfip.runtime.serving_study import StudyWorker
    volume.reload()
    return StudyWorker(model)


@app.cls(image=fast_image,gpu='B200',cpu=8,memory=98304,volumes={'/research':volume},
         max_containers=1,scaledown_window=10,timeout=180,startup_timeout=1200)
@modal.concurrent(max_inputs=64,target_inputs=64)
class Decider:
    @modal.enter()
    def load(self):
        self.worker=load_worker('decider')
    @modal.method()
    def configure(self,mode):
        return self.worker.configure(mode)
    @modal.method()
    def predict(self,record):
        return self.worker.predict(record)
    @modal.method()
    def finish(self):
        return self.worker.finish()


@app.cls(image=fast_image,gpu='L40S',cpu=4,memory=32768,volumes={'/research':volume},
         max_containers=1,scaledown_window=10,timeout=180,startup_timeout=1200)
@modal.concurrent(max_inputs=64,target_inputs=64)
class Clef:
    @modal.enter()
    def load(self):
        self.worker=load_worker('clef')
    @modal.method()
    def configure(self,mode):
        return self.worker.configure(mode)
    @modal.method()
    def predict(self,record):
        return self.worker.predict(record)
    @modal.method()
    def finish(self):
        return self.worker.finish()


@app.cls(image=sglang_image,gpu='B200',cpu=8,memory=98304,volumes={'/research':volume},
         max_containers=1,scaledown_window=10,timeout=180,startup_timeout=1200)
@modal.concurrent(max_inputs=64,target_inputs=64)
class Security:
    @modal.enter()
    def load(self):
        self.worker=load_worker('security')
    @modal.method()
    def configure(self,mode):
        return self.worker.configure(mode)
    @modal.method()
    def predict(self,record):
        return self.worker.predict(record)
    @modal.method()
    def finish(self):
        return self.worker.finish()
    @modal.exit()
    def close(self):
        self.worker.close()


async def phase(worker,rate,duration,phase_id):
    """Fixed open-loop arrivals; latency starts at scheduled arrival, not dispatch."""
    origin=time.perf_counter()+.5
    in_flight=0
    async def one(index):
        nonlocal in_flight
        scheduled=origin+index/rate
        await asyncio.sleep(max(0,scheduled-time.perf_counter()))
        dispatched=time.perf_counter()
        row={'index':index,'scheduled_seconds':index/rate,
             'dispatch_lag_ms':(dispatched-scheduled)*1000}
        if in_flight>=128:
            row.update(status='dropped',finished_seconds=dispatched-origin)
            return row
        in_flight+=1
        try:
            record=request_record(phase_id*10000+index)
            response=await asyncio.wait_for(worker.predict.remote.aio(record),timeout=120)
            finished=time.perf_counter()
            row.update(response,status='ok',latency_ms=(finished-scheduled)*1000,
                       finished_seconds=finished-origin)
        except Exception as exc:
            row.update(status='error',error=str(exc),finished_seconds=time.perf_counter()-origin)
        finally:
            in_flight-=1
        return row
    rows=await asyncio.gather(*(one(i) for i in range(math.ceil(rate*duration))))
    return {'summary':summarize_phase(rows,duration,rate),'samples':rows}


async def run_study(model,round_id,commit,duration,folder):
    worker={'decider':Decider,'clef':Clef,'security':Security}[model]()
    result={'protocol':SERVING_PROTOCOL,'model':model,'round':round_id,
            'source_commit':commit,'status':'running',
            'client':{'location':'external author workstation in Sweden',
                      'transport':'authenticated Modal RPC',
                      'max_in_flight':128,'server_max_concurrent_inputs':64,
                      'server_gpu_execution':'serialized for native/production; native_concurrent dispatches to SGLang scheduler'},
            'modes':{},'duration_per_rate_seconds':duration}
    modes=['native','production'] if model!='security' else ['native','native_concurrent']
    random.Random(20261012+round_id).shuffle(modes)
    result['mode_order']=modes
    dest=folder/f'{model}-serving-round-{round_id}.json'
    def save():
        dest.write_text(json.dumps(result,indent=2))
    try:
        for mode in modes:
            started=time.perf_counter()
            report={'setup':await worker.configure.remote.aio(mode),'phases':[]}
            report['client_setup_seconds']=time.perf_counter()-started
            result['modes'][mode]=report; save()
            for phase_id,rate in enumerate((5,15,30)):
                phase_report=await phase(worker,rate,duration,phase_id)
                report['phases'].append(phase_report); save()
                print(json.dumps({'model':model,'round':round_id,'mode':mode,
                                  'rate':rate,**phase_report['summary']}),flush=True)
                if phase_report['summary']['failed']:
                    raise RuntimeError('Serving phase returned errors; stop this container and preserve results')
            report['finish']=await worker.finish.remote.aio(); save()
        result['status']='complete'
    except Exception as exc:
        result['status']='failed'; result['failure']=str(exc)
    save()
    return result


@app.local_entrypoint()
def main(model:str='decider',round_id:int=0,commit:str='unknown',duration:int=60):
    folder=Path('benchmarks/results/strengthened');folder.mkdir(parents=True,exist_ok=True)
    asyncio.run(run_study(model,round_id,commit,duration,folder))
