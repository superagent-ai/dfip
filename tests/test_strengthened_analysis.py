import importlib.util
from pathlib import Path
import pytest
from dfip.study_protocol import summarize_phase

spec=importlib.util.spec_from_file_location('strengthened_analysis',Path(__file__).parents[1]/'scripts/analyze_strengthened.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def request(container='a'):
    return {'status':'ok','latency_ms':100,'dispatch_lag_ms':0,'finished_seconds':.5,
            'service_ms':50,'queue_ms':25,'container_id':container}


def phase(row):
    return {'samples':[row],'summary':summarize_phase([row],1,1)}


def test_container_replacement_cannot_be_reported_as_one_paired_serving_worker():
    record={'model':'clef','modes':{'native':{'phases':[phase(request('a'))]},
                                  'production':{'phases':[phase(request('b'))]}}}
    with pytest.raises(ValueError,match='one worker container'):
        analysis.serving_rows([record])


def test_tampered_summary_rejected_before_generating_serving_table():
    data=phase(request());data['summary']['p95_ms']=50
    with pytest.raises(ValueError,match='differs'):
        analysis.serving_rows([{'model':'clef','modes':{'native':{'phases':[data]}}}])


def test_stream_fidelity_catches_threshold_change_without_label_change():
    def mode(probability):
        return {'phases':[{'samples':[{'status':'ok','prediction':[
            {'id':'test:team','prediction':'other','probabilities':{'chosen':probability,'other':1-probability}}
        ]}]}]}
    report=analysis.continuous_fidelity({'modes':{'native':mode(.26),'production':mode(.24)}})
    assert report['argmax_flips']==0
    assert report['threshold_crossings']['0.25']==1
    assert report['matched_decisions']==1
    assert report['max_probability_drift']==pytest.approx(.02)
