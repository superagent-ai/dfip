"""Experimental Security-One head ablation; never used by deployment."""

import modal
from dfip.config import settings
from dfip.cloud.images import sglang_image as image, volume as research

app = modal.App(settings().app_name + "-sglang-benchmark")


@app.function(image=image, volumes={"/research": research}, timeout=120)
def probe():
    import subprocess, os

    env = os.environ.copy()
    env["PYTHONPATH"] = "/opt/dfip/dfip/runtime:/opt/dfip"
    env["DFIP_SGLANG_HEAD"] = "triton"
    env["DFIP_SGLANG_TOKEN_IDS"] = "[1,2]"
    return subprocess.run(
        [
            "/opt/sglang/bin/python",
            "-c",
            "import sys,sitecustomize;from sglang.srt.layers.logits_processor import LogitsProcessor;print(sys.path);print(sitecustomize.__file__);print(LogitsProcessor._compute_lm_head.__module__)",
        ],
        env=env,
        capture_output=True,
        text=True,
    ).__dict__


@app.function(
    image=image,
    gpu="B200",
    cpu=8,
    memory=98304,
    volumes={"/research": research},
    timeout=7200,
)
def benchmark(
    quality_limit: int = 400, repetitions: int = 100, output: str = "sglang-final"
):
    import json, sys, os, subprocess, time
    from pathlib import Path

    sys.path.insert(0, "/opt/dfip")
    import torch
    from transformers import AutoTokenizer
    from dfip.runtime.adapters import SecurityAdapter
    from dfip.runtime.benchmark import (
        dataset,
        scenarios,
        distributions,
        summarize_quality,
    )
    import httpx, numpy as np

    model_id = "superagent-ai/security-one-27b"
    meta = json.loads(
        Path("/research/prepared/superagent-ai__security-one-27b.json").read_text()
    )
    # The lightweight client shares the released prompt construction without loading a second model.
    client = SecurityAdapter.__new__(SecurityAdapter)
    client.tokenizer = AutoTokenizer.from_pretrained(meta["path"])
    client.temperature = 0.14527332485151376
    import itertools, string

    client.labels = []
    seen = set()
    for code in list(string.ascii_uppercase) + [
        "".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)
    ]:
        ids = client.tokenizer.encode(code, add_special_tokens=False)
        if (
            len(ids) == 1
            and ids[0] not in seen
            and client.tokenizer.decode(ids) == code
        ):
            client.labels.append((code, ids[0]))
            seen.add(ids[0])
        if len(client.labels) == 255:
            break
    transport = httpx.Client(base_url="http://127.0.0.1:30000", timeout=180)

    def predict(records):
        pairs = [
            client.render(r["state"], q)
            for r in records
            for q in r["questions"].values()
        ]
        input_ids = [
            client.tokenizer.encode(s, add_special_tokens=False) for s, _ in pairs
        ]
        label_ids = [[i for _, i in client.labels[: len(keys)]] for _, keys in pairs]
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "ignore_eos": True,
            },
            "stream": False,
            "return_logprob": True,
            "token_ids_logprob": label_ids,
            "logprob_start_len": -1,
            "top_logprobs_num": 0,
            "return_text_in_logprobs": False,
        }
        response = transport.post("/generate", json=payload)
        response.raise_for_status()
        outputs = response.json()
        outputs = outputs if isinstance(outputs, list) else [outputs]
        answers = []
        index = 0
        for r in records:
            item = {}
            for key, q in r["questions"].items():
                selected = outputs[index]["meta_info"]["output_token_ids_logprobs"][0]
                by_id = {int(v[1]): float(v[0]) for v in selected}
                values = (
                    np.array([by_id[i] for i in label_ids[index]]) / client.temperature
                )
                weights = np.exp(values - values.max())
                weights /= weights.sum()
                keys = pairs[index][1]
                p = dict(zip(keys, map(float, weights)))
                index += 1
                a = {"type": q["type"], "probabilities": p}
                if q["type"] == "noul":
                    a["noul"] = p["true"]
                elif q["type"] == "choice":
                    a["choice"] = max(p, key=p.get)
                else:
                    a["score"] = sum(int(k) * v for k, v in p.items())
                item[key] = a
            answers.append({"answers": item})
        return answers

    records = dataset()[:quality_limit]
    workloads = scenarios()
    folder = Path("/research/results") / output
    folder.mkdir(parents=True, exist_ok=True)
    result = {
        "model_id": model_id,
        "revision": meta["revision"],
        "hardware": torch.cuda.get_device_name(),
        "engine": "SGLang 0.5.19 BF16, TensorRT-LLM MHA, float32 Mamba state",
        "timing_scope": "in-container HTTP decision client, includes encoding, batching, scheduling, GPU execution and probability readout",
        "prefix_cache": "disabled for paired compute comparison",
        "quality_cases": len(records),
        "modes": {},
    }
    reference = None
    for mode in ("native", "dfip_cublas", "dfip_triton", "native_repeat"):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = "/opt/dfip/dfip/runtime:/opt/dfip"
        environment["DFIP_SGLANG_HEAD"] = (
            mode.removeprefix("dfip_") if mode.startswith("dfip_") else "native"
        )
        environment["DFIP_SGLANG_TOKEN_IDS"] = json.dumps([i for _, i in client.labels])
        log_path = folder / (mode + "-server.log")
        log = log_path.open("w")
        command = [
            "/opt/sglang/bin/python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            meta["path"],
            "--host",
            "127.0.0.1",
            "--port",
            "30000",
            "--language-only",
            "--dtype",
            "bfloat16",
            "--context-length",
            "8192",
            "--mem-fraction-static",
            ".80",
            "--max-running-requests",
            "32",
            "--max-total-tokens",
            "16384",
            "--max-mamba-cache-size",
            "32",
            "--mamba-ssm-dtype",
            "float32",
            "--attention-backend",
            "trtllm_mha",
            "--chunked-prefill-size",
            "8192",
            "--cuda-graph-max-bs-decode",
            "32",
            "--disable-radix-cache",
        ]
        start = time.perf_counter()
        process = subprocess.Popen(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            while time.perf_counter() - start < 900:
                if process.poll() is not None:
                    raise RuntimeError(log_path.read_text()[-10000:])
                try:
                    health = transport.get("/health", timeout=2)
                    if health.status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("SGLang startup exceeded 15 minutes")
            setup = time.perf_counter() - start
            if (
                mode.startswith("dfip_")
                and '"dfip_decision_head": true' not in log_path.read_text()
            ):
                raise RuntimeError(
                    "Decision head was not verified during startup: "
                    + log_path.read_text()[-4000:]
                )
            rows = []
            for i, r in enumerate(records):
                rows.extend(distributions([r], predict([r])))
                if (i + 1) % 50 == 0:
                    print(json.dumps({"mode": mode, "quality_done": i + 1}), flush=True)
            quality = summarize_quality(rows, reference)
            if reference is None:
                reference = rows
            (folder / (mode + "-predictions.json")).write_text(json.dumps(rows))
            timings = []
            for workload in workloads:
                first = time.perf_counter()
                predict(workload["records"])
                first = time.perf_counter() - first
                for _ in range(5):
                    predict(workload["records"])
                samples = []
                for _ in range(repetitions):
                    t = time.perf_counter()
                    predict(workload["records"])
                    samples.append((time.perf_counter() - t) * 1000)
                row = {
                    "scenario": workload["name"],
                    "decisions_per_call": workload["decisions"],
                    "p50_ms": float(np.median(samples)),
                    "p95_ms": float(np.percentile(samples, 95)),
                    "mean_ms": float(np.mean(samples)),
                    "decisions_per_second": workload["decisions"]
                    * 1000
                    / float(np.mean(samples)),
                    "first_call_seconds": first,
                    "samples_ms": samples,
                }
                timings.append(row)
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "scenario": row["scenario"],
                            "p50_ms": row["p50_ms"],
                        }
                    ),
                    flush=True,
                )
            result["modes"][mode] = {
                "quality": quality,
                "timings": timings,
                "server_start_seconds": setup,
            }
            (folder / "security-one.json").write_text(json.dumps(result, indent=2))
            research.commit()
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
            research.commit()
    return result


@app.local_entrypoint()
def main(
    quality: int = 400,
    repetitions: int = 100,
    output: str = "sglang-final",
    action: str = "benchmark",
):
    import json
    from pathlib import Path

    result = (
        probe.remote()
        if action == "probe"
        else benchmark.remote(quality, repetitions, output)
    )
    Path(
        "sglang-" + ("probe" if action == "probe" else "results") + ".json"
    ).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
