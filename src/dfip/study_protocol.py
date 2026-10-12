"""Pure request-generation and accounting for the sustained Modal study."""

import copy
import math
import statistics

SERVING_PROTOCOL = 'dfip-paper-v3-external-serving'


def request_record(index):
    """Changing contents, bounded recurring lengths and changing answer widths."""
    width = (2, 8, 16)[index % 3]
    state = ('Audit event recorded. ' * (index % 12 + 1)) + (
        'Please refund the duplicate payment.' if index % 2 == 0
        else 'The application crashes when I sign in.'
    ) + f' Ticket {index}.'
    question = {'type':'choice', 'instructions':'Which team should handle this request?',
                'criteria':{'billing':'Payments and refunds','technical':'Software bugs and errors'}}
    question['criteria'].update({f'team_{i}':f'Other department {i}' for i in range(2,width)})
    return {'id':f'serving-{index}', 'state':state, 'questions':{'team':question}}


def summarize_phase(rows, duration, rate):
    if duration <= 0 or rate <= 0 or len(rows) != math.ceil(duration*rate):
        raise ValueError('Offered-load accounting mismatch')
    ok=[r for r in rows if r['status']=='ok']
    latency=[r['latency_ms'] for r in ok]
    if any(x <= 0 or not math.isfinite(x) for x in latency):
        raise ValueError('Invalid latency sample')
    def quantile(values,p):
        if not values:
            return None
        ordered=sorted(values); pos=(len(ordered)-1)*p
        low=math.floor(pos); high=math.ceil(pos)
        return ordered[low]+(ordered[high]-ordered[low])*(pos-low)
    return {'offered':len(rows),'successful':len(ok),
            'dropped':sum(r['status']=='dropped' for r in rows),
            'failed':sum(r['status']=='error' for r in rows),
            'offered_rps':rate,'duration_seconds':duration,
            'completed_within_window':sum(r['status']=='ok' and r['finished_seconds']<=duration for r in rows),
            'throughput_within_window_rps':sum(r['status']=='ok' and r['finished_seconds']<=duration for r in rows)/duration,
            'drain_seconds':max(0,max((r['finished_seconds'] for r in rows),default=duration)-duration),
            'p50_ms':statistics.median(latency) if latency else None,
            'p95_ms':quantile(latency,.95),'p99_ms':quantile(latency,.99),
            'max_dispatch_lag_ms':max((r['dispatch_lag_ms'] for r in rows),default=0),
            'server_service_p50_ms':statistics.median(r['service_ms'] for r in ok) if ok else None,
            'server_queue_p95_ms':quantile([r['queue_ms'] for r in ok],.95)}


def records_for_fidelity(count=128):
    return [copy.deepcopy(request_record(i)) for i in range(count)]
