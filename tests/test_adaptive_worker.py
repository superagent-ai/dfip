from types import SimpleNamespace
import sys
import pytest
from dfip.adaptive_protocol import verify_worker, mixed_record
from dfip.runtime.adaptive_engine import AdaptiveDecisionEngine


@pytest.mark.parametrize('response', [
    {'container_id': 'replacement', 'worker_mode': 'adaptive'},
    {'container_id': 'original', 'worker_mode': 'native'},
    {'container_id': 'original'},
])
def test_worker_replacement_or_lost_configuration_is_rejected(response):
    with pytest.raises(ValueError, match='identity or configured mode'):
        verify_worker(response, 'original', 'adaptive')


def test_configured_worker_is_accepted():
    verify_worker({'container_id': 'original', 'worker_mode': 'adaptive'}, 'original', 'adaptive')


class Backend:
    def __init__(self):
        self.states = []
        self.adapter = SimpleNamespace(enable_graphs=self.states.append)
    def configure(self, mode):
        self.mode = mode
    def predict(self, records):
        return [{'id': r.get('id'), 'answers': {'team': {'choice': 'billing'}}} for r in records]
    def diagnostics(self):
        return {'graphs': []}


def test_opt_in_api_preserves_answers_and_limits_replay_to_tested_schema(monkeypatch):
    monkeypatch.setitem(sys.modules, 'dfip.runtime.benchmark', SimpleNamespace(sync=lambda: None))
    backend = Backend()
    engine = AdaptiveDecisionEngine('clef', backend=backend)
    record = mixed_record(0)
    assert engine.predict([record]) == backend.predict([record])
    assert backend.states == [True, True]
    backend.states.clear()
    engine.predict([record, mixed_record(1)])
    assert backend.states == [False, True]
    backend.states.clear()
    record['questions'] = {'check': {'type': 'noul', 'instructions': 'Does this request need a refund?'}}
    engine.predict([record])
    assert backend.states == [False, True]
    assert engine.diagnostics()['optimization']['experimental'] is True


def test_opt_in_api_restores_graph_state_after_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'dfip.runtime.benchmark', SimpleNamespace(sync=lambda: None))
    backend = Backend()
    def fail(records):
        raise RuntimeError('GPU fault')
    backend.predict = fail
    engine = AdaptiveDecisionEngine('decider', backend=backend)
    with pytest.raises(RuntimeError, match='GPU fault'):
        engine.predict([mixed_record(0)])
    assert backend.states == [True, True]


def test_other_checkpoint_families_cannot_enable_adaptive_replay():
    with pytest.raises(ValueError, match='Clef and Decider'):
        AdaptiveDecisionEngine('liquid', backend=Backend())
