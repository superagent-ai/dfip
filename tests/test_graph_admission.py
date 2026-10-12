import pytest
from dfip.graph_admission import AdaptiveAdmission


def test_recurring_diversity_below_hotness_threshold_never_recaptures():
    policy = AdaptiveAdmission()
    for i in range(5000):
        assert policy.select(i % 12, cached=False, full=False).startswith('native_')
    assert policy.attempts == 0
    assert len(policy.recent) == 128
    assert len(policy.counts) == 12


def test_hot_shape_promotes_once_and_replays_with_bounded_native_tail():
    policy = AdaptiveAdmission()
    cached = set()
    for i in range(1000):
        key = 'hot' if i % 5 < 3 else i
        action = policy.select(key, cached=key in cached, full=len(cached) >= 4)
        if action == 'capture':
            cached.add(key)
    assert cached == {'hot'}
    assert policy.attempts == 1
    assert policy.actions['replay'] > 500
    assert len(policy.counts) <= 128


def test_full_resident_set_never_evicts_for_new_frequent_shapes():
    policy = AdaptiveAdmission(max_captures=1)
    for _ in range(32):
        action = policy.select('first', cached=False, full=False)
    assert action == 'capture'
    for _ in range(1000):
        assert policy.select('second', cached=False, full=True) == 'native_budget'
    assert policy.select('first', cached=True, full=True) == 'replay'
    assert policy.attempts == 1


def test_failed_attempt_consumes_budget_even_when_cache_is_empty():
    policy = AdaptiveAdmission(max_captures=1)
    for _ in range(32):
        policy.select('failing', cached=False, full=False)
    assert policy.select('another', cached=False, full=False) == 'native_budget'


def test_fidelity_checks_can_freeze_admission_without_disabling_replay():
    policy = AdaptiveAdmission()
    policy.frozen = True
    assert policy.select('new', cached=False, full=False) == 'native_budget'
    assert policy.select('resident', cached=True, full=False) == 'replay'


def test_invalid_admission_limits_rejected():
    with pytest.raises(ValueError):
        AdaptiveAdmission(window=4)
