import pytest
from dfip.study_protocol import request_record, summarize_phase
from dfip.study_protocol import compare_reported_answers


def test_serving_payloads_change_content_and_candidate_width():
    rows=[request_record(i) for i in range(24)]
    assert len({r['state'] for r in rows})==24
    assert {len(r['questions']['team']['criteria']) for r in rows}=={2,8,16}


def test_overload_accounting_includes_drops_and_post_window_drain():
    rows=[{'status':'ok','latency_ms':100,'dispatch_lag_ms':0,'finished_seconds':.5,
           'service_ms':50,'queue_ms':25},
          {'status':'ok','latency_ms':2000,'dispatch_lag_ms':10,'finished_seconds':2,
           'service_ms':60,'queue_ms':1500},
          {'status':'dropped','dispatch_lag_ms':3,'finished_seconds':.9}]
    summary=summarize_phase(rows,1,3)
    assert summary['successful']==2 and summary['dropped']==1
    assert summary['throughput_within_window_rps']==1
    assert summary['drain_seconds']==1
    assert summary['p50_ms']==1050
    assert summary['p99_ms']>1900


def test_missing_offered_request_cannot_silently_lower_tail_latency():
    with pytest.raises(ValueError,match='accounting'):
        summarize_phase([],60,5)


def test_direct_api_audit_catches_choice_change_hidden_by_rounded_tie():
    reference=[{'id':'r:team','api_choice':'billing','api_score':1.2345}]
    actual=[{'id':'r:team','api_choice':'technical','api_score':1.2}]
    report=compare_reported_answers(reference,actual)
    assert report['choice_changes']==1
    assert report['choice_fields_compared']==1
    assert report['max_score_drift']==pytest.approx(.0345)


def test_missing_api_fields_do_not_establish_zero_reported_score_drift():
    report=compare_reported_answers([{'id':'r'}],[{'id':'r'}])
    assert report['score_fields_compared']==0
    assert report['max_score_drift'] is None
