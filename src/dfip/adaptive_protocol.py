"""Requests for a separate exploratory adaptive-replay serving pilot."""

from dfip.study_protocol import request_record

PROTOCOL = 'dfip-paper-v4-adaptive-pilot'


def mixed_record(index):
    """60% recurring short shape with changing tickets; 40% original diversity.

    This intentional hot/long-tail mix is a diagnostic workload, not observed
    production popularity. Inputs change and predictions are never cached.
    """
    if index % 5 < 3:
        record = request_record(index * 36)
        record['id'] = f'serving-{index}'
        return record
    return request_record(index)
