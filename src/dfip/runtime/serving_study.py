"""Resident worker state; mutable CUDA buffers are always serialized."""

import threading
import time
import torch
from dfip.runtime.research import AblationEngine, manifest, compare
from dfip.runtime.benchmark import distributions, sync
from dfip.models import get_model
from dfip.study_protocol import records_for_fidelity


class StudyWorker:
    def __init__(self, model):
        self.model = model
        torch.set_num_threads(8 if get_model(model).gpu == 'B200' else 4)
        self.lock = threading.RLock()
        start = time.perf_counter()
        if model == 'security':
            from dfip.runtime.sglang_runtime import SGLangDecisionEngine
            self.engine = SGLangDecisionEngine(head_mode='native')
        else:
            self.engine = AblationEngine(get_model(model).model_id)
            self.engine.configure('native')
        sync()
        self.load_seconds = time.perf_counter()-start
        self.environment = manifest()
        self.mode = 'native'
        self.references = []
        for record in records_for_fidelity():
            self.references.extend(distributions([record],self.engine.predict([record])))
        self.calls = 0

    def configure(self, mode):
        with self.lock:
            if self.model == 'security' and mode != 'native':
                raise ValueError('Security-One selected serving policy remains native')
            if self.model != 'security':
                self.engine.configure(mode)
            self.mode = mode
            self.calls = 0
            from dfip.study_protocol import request_record
            admission=[]
            for i in range(48):
                start=time.perf_counter()
                self.engine.predict([request_record(i)])
                sync(); admission.append((time.perf_counter()-start)*1000)
            predictions=[]
            for r in records_for_fidelity():
                predictions.extend(distributions([r],self.engine.predict([r])))
            backend={}
            if self.model in ('clef','decider'):
                import transformers.models.qwen3_5.modeling_qwen3_5 as q
                backend['causal_conv1d_available']=q.causal_conv1d_fn is not None
                backend['chunk_gated_delta_rule_available']=q.chunk_gated_delta_rule is not None
            if mode=='native' and self.model!='security':
                try:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
                        self.engine.predict([records_for_fidelity(1)[0]])
                        sync()
                    backend['cuda_kernel_names']=sorted({e.name for e in profile.events()
                                                        if str(e.device_type).endswith('CUDA')})
                except Exception as exc:
                    backend['profile_error']=str(exc)
            return {'mode':mode,'environment':self.environment,'load_seconds':self.load_seconds,
                    'admission_ms':admission,'initial_fidelity':compare(self.references,predictions),
                    'native_backend':backend,
                    'diagnostics':self.engine.diagnostics()}

    def predict(self, record):
        received=time.perf_counter()
        with self.lock:
            started=time.perf_counter()
            output=self.engine.predict([record])
            sync()
            finished=time.perf_counter()
            self.calls+=1
            return {'request_id':record['id'], 'container_id':self.environment['container_id'],
                    'queue_ms':(started-received)*1000,'service_ms':(finished-started)*1000,
                    'prediction':distributions([record],output)}

    def finish(self):
        with self.lock:
            predictions=[]
            for r in records_for_fidelity():
                predictions.extend(distributions([r],self.engine.predict([r])))
            sync()
            return {'mode':self.mode,'calls':self.calls,
                    'final_fidelity':compare(self.references,predictions),
                    'diagnostics':self.engine.diagnostics(),
                    'allocated_bytes':torch.cuda.memory_allocated(),
                    'reserved_bytes':torch.cuda.memory_reserved(),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated()}

    def close(self):
        if hasattr(self.engine,'close'):
            self.engine.close()
