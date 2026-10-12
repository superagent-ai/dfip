from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import pytest
from dfip.native_dispatch import dispatch_lock
from dfip.adaptive_protocol import mixed_record


def test_native_dispatch_allows_overlapping_requests_without_shared_results():
    lock = dispatch_lock('native', concurrent=True)
    barrier = Barrier(2)
    def request(identity):
        with lock:
            barrier.wait(timeout=2)
            return {'id': identity}
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(request, 'a'); b = pool.submit(request, 'b')
        assert a.result() == {'id': 'a'}
        assert b.result() == {'id': 'b'}


def test_mutated_head_cannot_enable_concurrent_dispatch():
    with pytest.raises(ValueError, match='native SGLang head'):
        dispatch_lock('triton', concurrent=True)


def test_hot_mix_changes_inputs_and_retains_the_prescribed_popularity():
    records = [mixed_record(i) for i in range(100)]
    hot = [r for i, r in enumerate(records) if i % 5 < 3]
    assert len(hot) == 60
    assert len({r['state'] for r in hot}) == 60
    assert len({r['id'] for r in records}) == 100
    assert all(len(r['questions']['team']['criteria']) == 2 for r in hot)
