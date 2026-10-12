"""Released Security-One prompts and conditional readout over a local SGLang server."""

import itertools
import json
import os
from pathlib import Path
import string
import subprocess
import time
from contextlib import nullcontext

import httpx
import numpy as np
from transformers import AutoTokenizer
from dfip.runtime.adapters import SecurityAdapter, checkpoint
from dfip.native_dispatch import dispatch_lock


class SGLangDecisionEngine:
    def __init__(self, head_mode="native", concurrent=False):
        self.lock = dispatch_lock(head_mode, concurrent)
        self.client = SecurityAdapter.__new__(SecurityAdapter)
        path = checkpoint("superagent-ai/security-one-27b")
        self.client.tokenizer = AutoTokenizer.from_pretrained(path)
        self.client.temperature = 0.14527332485151376
        self.client.labels = []
        seen = set()
        codes = list(string.ascii_uppercase) + [
            "".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)
        ]
        for code in codes:
            ids = self.client.tokenizer.encode(code, add_special_tokens=False)
            if (
                len(ids) == 1
                and ids[0] not in seen
                and self.client.tokenizer.decode(ids) == code
            ):
                self.client.labels.append((code, ids[0]))
                seen.add(ids[0])
            if len(self.client.labels) == 255:
                break
        self.transport = httpx.Client(base_url="http://127.0.0.1:30000", timeout=180)
        self.mode = head_mode
        environment = os.environ.copy()
        environment["PYTHONPATH"] = (
            "/opt/dfip" if head_mode == "native" else "/opt/dfip/dfip/runtime:/opt/dfip"
        )
        environment["DFIP_SGLANG_HEAD"] = head_mode
        environment["DFIP_SGLANG_TOKEN_IDS"] = json.dumps(
            [i for _, i in self.client.labels]
        )
        self.log_path = Path("/tmp/dfip-security-server.log")
        self.log = self.log_path.open("w")
        command = [
            "/opt/sglang/bin/python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            path,
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
        self.process = subprocess.Popen(
            command, env=environment, stdout=self.log, stderr=subprocess.STDOUT
        )
        try:
            while time.perf_counter() - start < 900:
                if self.process.poll() is not None:
                    raise RuntimeError(self.log_path.read_text()[-5000:])
                try:
                    if self.transport.get("/health", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("Security-One server startup exceeded 15 minutes")
        except Exception:
            self.close()
            raise
        self.start_seconds = time.perf_counter() - start

    def predict(self, records):
        from dfip.schema import validate_records

        validate_records(records)
        if not isinstance(records, list) or not records:
            raise ValueError("Provide a nonempty list of records")
        for r in records:
            if (
                "state" not in r
                or not isinstance(r.get("questions"), dict)
                or not r["questions"]
            ):
                raise ValueError("Each record needs state and questions")
            if r.get("images") or r.get("videos"):
                raise ValueError("This release covers text decisions")
            for q in r["questions"].values():
                if q.get("type") not in ("choice", "noul", "score"):
                    raise ValueError("Unsupported question type")
                if q["type"] != "noul" and not q.get("criteria"):
                    raise ValueError("Criteria must be nonempty")
        with self.lock:
            pairs = [
                self.client.render(r["state"], q)
                for r in records
                for q in r["questions"].values()
            ]
            if any(not 1 <= len(keys) <= 255 for _, keys in pairs):
                raise ValueError("Supported range is 1–255 options")
            inputs = [
                self.client.tokenizer.encode(s, add_special_tokens=False)
                for s, _ in pairs
            ]
            if any(len(row) > 8192 for row in inputs):
                raise ValueError("Context exceeds 8192 tokens; no truncation")
            labels = [
                [i for _, i in self.client.labels[: len(keys)]] for _, keys in pairs
            ]
            payload = {
                "input_ids": inputs,
                "sampling_params": {
                    "max_new_tokens": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "ignore_eos": True,
                },
                "stream": False,
                "return_logprob": True,
                "token_ids_logprob": labels,
                "logprob_start_len": -1,
                "top_logprobs_num": 0,
                "return_text_in_logprobs": False,
            }
            response = self.transport.post("/generate", json=payload)
            response.raise_for_status()
            outputs = response.json()
            outputs = outputs if isinstance(outputs, list) else [outputs]
            result = []
            index = 0
            for r in records:
                answers = {}
                for key, q in r["questions"].items():
                    by_id = {
                        int(v[1]): float(v[0])
                        for v in outputs[index]["meta_info"][
                            "output_token_ids_logprobs"
                        ][0]
                    }
                    z = (
                        np.array([by_id[i] for i in labels[index]])
                        / self.client.temperature
                    )
                    p = np.exp(z - z.max())
                    p /= p.sum()
                    values = dict(zip(pairs[index][1], map(float, p)))
                    index += 1
                    answer = {"type": q["type"], "probabilities": values}
                    if q["type"] == "noul":
                        answer["noul"] = values["true"]
                    elif q["type"] == "choice":
                        answer["choice"] = max(values, key=values.get)
                    else:
                        answer["score"] = sum(int(k) * v for k, v in values.items())
                    answers[key] = answer
                result.append({"answers": answers})
            return result

    def diagnostics(self):
        return {
            "model_id": "superagent-ai/security-one-27b",
            "engine": "SGLang 0.5.19",
            "head_mode": self.mode,
            "server_start_seconds": self.start_seconds,
            "prefix_cache": False,
            "request_dispatch": "concurrent" if isinstance(self.lock, nullcontext) else "serialized",
        }

    def close(self):
        if hasattr(self, "process"):
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if hasattr(self, "transport"):
            self.transport.close()
        if hasattr(self, "log"):
            self.log.close()
