"""Frozen exploratory evaluation of one capability-based batching policy."""
import hashlib
import json
import random
import statistics
import time

from dfip.contract_protocol import compare, record, validate_output
from dfip.runtime.contract_study import timed
from dfip.shared import BatchLimits, SharedBatcher, native_batches

CALIBRATION_STATES = (
    "My invoice has a duplicate charge. Please refund it.",
    "The application crashes whenever I sign in.",
    "Where is my order? It should have arrived yesterday.",
    "I want to buy a subscription for my team.",
    "Please cancel my account and erase the stored information.",
    "I cannot decide whether this is a billing problem or a software issue.",
    "No action is required. I am just checking the opening hours.",
    "My card was stolen and an unknown person made purchases.",
)
TEST_STATES = (
    "The receipt total is wrong, but the payment seems correct. Please investigate.",
    "An order arrived with a broken screen. I would prefer a replacement to a refund.",
    "I changed my address. Can you update the delivery details?",
    "Our production workspace is unavailable and all staff are locked out.",
    "Could someone explain the difference between your two plans?",
    "I received a suspicious password reset message that I did not request.",
    "Thank you, the previous problem is solved. There is no further request.",
    "Both my invoice and the login page look unusual; I am uncertain what is wrong.",
)


def fixtures(calibration=False, homogeneous=False, count=96):
    states = CALIBRATION_STATES if calibration else TEST_STATES
    rows = []
    for i in range(count):
        kind = "choice" if homogeneous else ("choice", "choice", "choice", "choice", "noul", "score")[i % 6]
        width = 4 if homogeneous else (2, 4, 8, 16)[i % 4]
        repeats = (1, 2, 6, 18)[(i//6) % 4]
        row = record((1000 if calibration else 10000) + i, kind, width, repeats)
        row["state"] = f"Case {row['id']}. " + (states[(i//6) % len(states)] + " ")*repeats
        if not calibration and not homogeneous and i % 11 == 0 and kind == "choice":
            # Ordered schemas are part of the contract. An uncalibrated GLiNER
            # schema must remain a singleton rather than borrow another gate.
            q = row["questions"]["decision"]
            q["criteria"] = dict(reversed(list(q["criteria"].items())))
        rows.append(row)
    return rows


def calibrate(adapter):
    rows = fixtures(calibration=True, count=96)
    groups = {}
    for row in rows:
        groups.setdefault(adapter.key(row), []).append(row)
    admitted = {}; evidence = []
    for key, group in groups.items():
        # One empirical gate per schema and size, with all available kinds and
        # varied lengths. No model name affects candidate selection.
        for size in (2, 4, 8):
            if len(group) < size:
                continue
            comparisons = []
            chunks = [group[i:i+size] for i in range(0, len(group), size) if len(group[i:i+size]) == size]
            for chunk in chunks:
                reference = [adapter.single(r) for r in chunk]
                actual = adapter.batch(chunk)
                for output in actual:
                    validate_output(output)
                comparisons.append(compare(reference, actual))
            representative = chunks[len(chunks)//2]
            for _ in range(2):
                adapter.batch(representative)
            serial = []; batched = []
            for j in range(3):
                calls = [(serial, lambda: [adapter.single(r) for r in representative]),
                         (batched, lambda: adapter.batch(representative))]
                if j % 2:
                    calls.reverse()
                for samples, call in calls:
                    samples.append(timed(call)[1])
            gain = statistics.median(serial)/statistics.median(batched)
            passes = all(c["passes"] for c in comparisons) and gain >= 1.10
            if passes:
                admitted.setdefault(key, []).append(size)
            evidence.append({"key": key, "size": size, "records": [r for chunk in chunks for r in chunk],
                             "comparisons": comparisons, "serial_ms": serial, "batch_ms": batched,
                             "median_gain": gain, "admitted": passes})
    return admitted, evidence


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered)-1, int(fraction*len(ordered))-1))]


def serving(adapter, scheduler, rows, arrivals, mode, reference):
    """Real-time resident replay; latency starts at the scheduled arrival.

    Fixture generation and reference computation are outside the replay.
    Length estimation, queueing and native calls occur on the replay path.
    HTTP/network transport is outside this experiment.
    """
    max_queue = 128; deadline_s = 2.0; horizon = arrivals[-1]
    pending = []; cursor = 0; completed = []; drops = []
    batches = []; start = time.perf_counter(); max_depth = 0
    while cursor < len(rows) or pending:
        now = time.perf_counter()-start
        while cursor < len(rows) and arrivals[cursor] <= now:
            if len(pending) >= max_queue:
                drops.append({"index": cursor, "reason": "queue capacity"})
            else:
                # Include tokenizer/parameter preparation in scheduled latency.
                length = scheduler.length(rows[cursor]) if mode == "shared" else None
                pending.append((cursor, length))
            cursor += 1
        expired = [i for i, _ in pending if now-arrivals[i] > deadline_s]
        drops.extend({"index": i, "reason": "queue deadline"} for i in expired)
        pending = [(i, n) for i, n in pending if i not in set(expired)]
        max_depth = max(max_depth, len(pending))
        if not pending:
            if cursor < len(rows):
                time.sleep(max(0, arrivals[cursor]-(time.perf_counter()-start)))
            continue
        head = pending[0][0]; key = adapter.key(rows[head])
        cap = max((1, *scheduler.admitted.get(key, ()))) if mode == "shared" else (1 if mode == "serial" else 8)
        compatible = sum(adapter.key(rows[i]) == key for i, _ in pending)
        remaining_wait = scheduler.limits.wait_ms/1000 - (now-arrivals[head])
        if cap > 1 and compatible < cap and remaining_wait > 0:
            time.sleep(min(remaining_wait, max(0, arrivals[cursor]-now)) if cursor < len(rows) else remaining_wait)
            continue
        if mode == "shared":
            positions = scheduler.select([rows[i] for i, _ in pending], [n for _, n in pending])
        else:
            positions = [p for p, (i, _) in enumerate(pending) if adapter.key(rows[i]) == key][:cap]
        indices = [pending[p][0] for p in positions]
        selected = [rows[i] for i in indices]
        values, execution_ms = timed(lambda: scheduler.execute(selected))
        if len(values) != len(indices):
            raise RuntimeError("Native serving batch lost request correspondence")
        finished = time.perf_counter()-start
        for i, output in zip(indices, values):
            validate_output(output)
            completed.append({"index": i, "latency_ms": (finished-arrivals[i])*1000,
                              "finish_s": finished, "output": output,
                              "comparison": compare(reference[i], output)})
        batches.append({"indices": indices, "execution_ms": execution_ms})
        chosen = set(positions)
        pending = [entry for p, entry in enumerate(pending) if p not in chosen]
    elapsed = time.perf_counter()-start
    latencies = [c["latency_ms"] for c in completed]
    return {"mode": mode, "offered": len(rows), "arrival_horizon_s": horizon,
            "completed_within_window": sum(c['finish_s'] <= horizon for c in completed),
            "completed_after_drain": len(completed), "elapsed_s": elapsed, "drops": drops,
            "errors": 0, "max_queue_depth": max_depth, "p50_ms": percentile(latencies, .50),
            "p95_ms": percentile(latencies, .95), "p99_ms": percentile(latencies, .99),
            "completed": completed, "batches": batches,
            "batched_fraction": sum(len(b['indices']) for b in batches if len(b['indices']) > 1)/len(rows),
            "limitation": "Short resident replay with warmed reference fixtures; no HTTP transport or production reliability claim"}


def shared(adapter, result, round_id):
    result['protocol'] = 'dfip-shared-contract-batching-v5-exploratory'
    start = time.perf_counter()
    admitted, evidence = calibrate(adapter)
    result['admission'] = {'seconds': time.perf_counter()-start, 'keys': admitted, 'evidence': evidence}
    limits = BatchLimits()
    scheduler = SharedBatcher(adapter, admitted, limits)
    result['policy'] = dict(vars(limits), min_calibration_gain=1.10, numeric_tolerance=.001,
                           score_tolerance=.01, threshold_grid=[.1, .25, .5, .75, .9],
                           guarantee='finite calibration only; no universal future-input equivalence guarantee')
    result['resident_shared'] = {}; result['serving_shared'] = {}
    rng = random.Random(20261015+round_id)
    for homogeneous in (True, False):
        name = 'homogeneous' if homogeneous else 'diverse'
        rows = fixtures(homogeneous=homogeneous)
        reference = [adapter.single(r) for r in rows]
        for output in reference:
            validate_output(output)
        calls = {'serial': lambda: ([adapter.single(r) for r in rows], [[i] for i in range(len(rows))]),
                 'native_batch': lambda: native_batches(adapter, rows), 'shared': lambda: scheduler.run(rows)}
        measurements = {mode: [] for mode in calls}; audits = {}; orders = []
        for mode, call in calls.items():
            (outputs, batches), _ = timed(call)
            audits[mode] = {'comparison': compare(reference, outputs), 'outputs': outputs, 'batches': batches}
        for _ in range(5):
            order = list(calls); rng.shuffle(order); orders.append(order)
            for mode in order:
                measurements[mode].append(timed(calls[mode])[1])
        result['resident_shared'][name] = {'records': rows, 'reference': reference, 'audits': audits,
                                          'samples_ms': measurements, 'order': orders,
                                          'median_ms': {m: statistics.median(s) for m, s in measurements.items()}}
        # 256 changing requests, paired steady or burst arrivals. Replay is
        # exploratory; intended 30-minute independent trials remain future work.
        trace = fixtures(homogeneous=homogeneous, count=256)
        ref = [adapter.single(r) for r in trace]
        arrivals = []; clock = 0.0
        for i in range(len(trace)):
            rate = 25 if homogeneous else (80 if (i//32) % 2 == 0 else 20)
            arrivals.append(clock); clock += 1/rate
        order = list(calls); rng.shuffle(order)
        result['serving_shared'][name] = {'records': trace, 'arrivals_s': arrivals, 'order': order,
                                         'payload_sha256': hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest(),
                                         'modes': {mode: serving(adapter, scheduler, trace, arrivals, mode, ref) for mode in order}}
    if adapter.model is None:
        from dfip.runtime.contract_study import profile
        result['profile'] = profile(adapter, fixtures(count=1)[0])
