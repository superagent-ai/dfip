"""Native admission and component profiling, prior to shared policy design."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
import traceback

import torch

from dfip.contract_models import contract_model
from dfip.contract_protocol import compare, conformance_records, record, validate_output
from dfip.runtime.contract_adapters import load_contract


def environment():
    packages = {}
    for name in ("torch", "transformers", "triton", "gliner2", "peft", "vllm", "fla-core"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"packages": packages, "python": __import__('sys').version, "cuda": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
            "worker": os.environ.get("MODAL_TASK_ID"), "gpu": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()}


def timed(fn):
    torch.cuda.synchronize(); started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter()-started)*1000


def profile(adapter, sample):
    if adapter.model is None:
        import gzip
        from collections import defaultdict
        started = time.perf_counter(); length = adapter.length(sample)
        _, request_ms = timed(lambda: adapter.single(sample))
        encode_ms = (time.perf_counter()-started)*1000-request_ms
        adapter.llm.start_profile()
        adapter.single(sample)
        adapter.llm.stop_profile()
        files = []; kernels = defaultdict(lambda: [0, 0.0])
        for path in adapter.profile_dir.rglob("*.pt.trace.json.gz"):
            files.append({"path": str(path), "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            with gzip.open(path, "rt") as stream:
                trace = json.load(stream)
            for event in trace.get("traceEvents", []):
                if event.get("cat") == "kernel" and event.get("ph") == "X":
                    entry = kernels[event['name']]; entry[0] += 1; entry[1] += event.get('dur', 0)
        return {"kind": "native vLLM worker Torch CUDA trace", "encode_and_parameter_prepare_ms": encode_ms,
                "request_ms_without_profiling": request_ms, "estimated_tokens": length, "files": files,
                "kernels": [{"name": name, "calls": values[0], "duration_us": values[1]}
                            for name, values in sorted(kernels.items(), key=lambda pair: pair[1][1], reverse=True)[:30]],
                "limitation": "One separately profiled warmed request; profiler timings are not benchmark latencies"}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True, profile_memory=True) as profiler:
        adapter.single(sample); torch.cuda.synchronize()
    events = sorted(profiler.key_averages(), key=lambda e: e.self_device_time_total, reverse=True)
    return {"kind": "Torch CPU/CUDA operator profile", "operators": [
        {"operator": e.key, "calls": e.count, "self_cpu_us": e.self_cpu_time_total,
         "self_device_us": e.self_device_time_total, "device_memory_bytes": e.device_memory_usage}
        for e in events[:30]], "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def native(adapter, result, round_id):
    records = conformance_records()
    result["conformance_payload_sha256"] = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    result["conformance"] = []
    for r in records:
        expected, ms = timed(lambda: adapter.single(r)); validate_output(expected)
        actual = adapter.single(r); validate_output(actual)
        report = compare(expected, actual)
        result["conformance"].append({"id": r["id"], "record": r, "reference": expected,
                                      "repeat": actual, "comparison": report, "native_ms": ms})
    cells = {"choice_short": [record(100)], "boolean_short": [record(101, "noul")],
             "score_short": [record(102, "score")], "choice_medium": [record(103, repeats=15)],
             "choice_long": [record(104, repeats=60)], "three_questions": [records[-1]],
             "four_requests": [record(i) for i in range(110,114)],
             "eight_requests": [record(i) for i in range(120,128)]}
    names = list(cells); random.Random(20261014+round_id).shuffle(names)
    result["cell_order"] = names; result["resident"] = {}
    for name in names:
        rows = cells[name]
        call = lambda: [adapter.single(r) for r in rows]
        for _ in range(3):
            call()
        samples = [timed(call)[1] for _ in range(20)]
        result["resident"][name] = {"samples_ms": samples, "median_ms": statistics.median(samples),
                                    "records": rows, "estimated_lengths": [adapter.length(r) for r in rows],
                                    "mode": "released native single-request API; multi-record cells serial"}
    result["profile"] = profile(adapter, record(200, repeats=7))
    # Direct native batching probes expose capability, drift and padding cost.
    result["batch_probes"] = []
    for size in (2,4,8):
        rows = [record(300+i, repeats=(1,3,7,15)[i%4]) for i in range(size)]
        reference = [adapter.single(r) for r in rows]
        for _ in range(2):
            adapter.batch(rows)
        outputs, batch_ms = timed(lambda: adapter.batch(rows))
        _, serial_ms = timed(lambda: [adapter.single(r) for r in rows])
        result["batch_probes"].append({"size": size, "records": rows, "reference": reference,
                                       "outputs": outputs, "comparison": compare(reference, outputs),
                                       "serial_ms": serial_ms, "batch_ms": batch_ms,
                                       "speedup": serial_ms/batch_ms})


def study(alias, action, commit, round_id):
    torch.set_num_threads(4)
    spec = contract_model(alias)
    result = {"protocol": "dfip-contract-native-admission-v5-exploratory", "action": action,
              "source_commit": commit, "model_id": spec.model_id, "revision": spec.revision,
              "alias": alias, "round": round_id, "status": "started", "environment": environment(),
              "stage": "load", "started_unix": time.time()}
    try:
        started = time.monotonic(); adapter = load_contract(alias)
        result["load_seconds"] = time.monotonic()-started
        result["capabilities"] = adapter.capabilities
        if hasattr(adapter, "runtime"):
            result["native_runtime"] = adapter.runtime.runtime_info()
        result["stage"] = "native admission and profiling" if action == "native" else "shared admission and evaluation"
        if action == "native":
            native(adapter, result, round_id)
        elif action == "shared_compiled":
            from dfip.runtime.shared_study import compiled_followup
            compiled_followup(adapter, result, round_id)
        else:
            from dfip.runtime.shared_study import shared
            shared(adapter, result, round_id)
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"; result["error"] = str(error); result["traceback"] = traceback.format_exc()
    result["finished_unix"] = time.time()
    folder = Path("/research/results/contracts"); folder.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"model": alias, "status": result["status"], "stage": result["stage"]}), flush=True)
    return result
