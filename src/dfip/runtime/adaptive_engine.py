"""Experimental request API; use the native app for conservative defaults."""
import threading
from dfip.models import get_model
from dfip.schema import validate_records


class AdaptiveDecisionEngine:
    def __init__(self, model_id, backend=None):
        model = get_model(model_id)
        if model.alias not in ('clef', 'decider'):
            raise ValueError('Adaptive replay currently covers Clef and Decider only')
        if backend is None:
            import torch
            from dfip.runtime.research import AblationEngine
            from dfip.study_protocol import records_for_fidelity
            from dfip.runtime.benchmark import sync
            torch.set_num_threads(8 if model.gpu == 'B200' else 4)
            backend = AblationEngine(model.model_id)
            backend.configure('native')
            for record in records_for_fidelity():
                backend.predict([record])
            sync()
        self.backend = backend
        self.lock = threading.RLock()
        self.backend.configure('adaptive')

    def predict(self, records):
        validate_records(records)
        from dfip.runtime.benchmark import sync
        with self.lock:
            eligible = (len(records) == 1 and len(records[0]['questions']) == 1
                        and next(iter(records[0]['questions'].values()))['type'] == 'choice')
            self.backend.adapter.enable_graphs(eligible)
            try:
                output = self.backend.predict(records)
                sync()
                return output
            finally:
                self.backend.adapter.enable_graphs(True)

    def diagnostics(self):
        with self.lock:
            return {'optimization': {'mode': 'adaptive', 'experimental': True,
                                    'policy': 'bounded resident graphs with native fallback'},
                    **self.backend.diagnostics()}
