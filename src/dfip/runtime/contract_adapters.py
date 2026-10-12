"""Released native paths with an explicit cross-request batching capability.

These research adapters retain native answer fields. GLiNER exposes label and
confidence rather than pretending its classification labels are ordinal scores.
"""
import json
import math
from pathlib import Path
import sys

import torch

from dfip.contract_models import contract_model


def prepared_path(alias):
    spec = contract_model(alias)
    metadata = json.loads((Path("/research/prepared/contracts") / (alias + ".json")).read_text())
    if metadata["revision"] != spec.revision:
        raise RuntimeError("Prepared model revision differs from research registry")
    return metadata["path"]


class OmniContract:
    capabilities = {"batch": True, "batch_key": "any text schema", "readout": "marker Transformer head",
                    "distribution": "choice and score; Boolean affirmative scalar", "dtype": "float16"}

    def __init__(self, path):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch.float16).to("cuda").eval()
        self.tokenizer = self.model.tokenizer

    def key(self, record):
        return "text"

    def length(self, record):
        return len(self.tokenizer.encode(record["state"])) + len(json.dumps(record["questions"]))

    def single(self, record):
        return self.model.system_one(record["state"], record["questions"])

    def batch(self, records):
        return self.model.system_one_batch([(r["state"], r["questions"]) for r in records])


class GlinerContract:
    capabilities = {"batch": True, "batch_key": "identical ordered classification schema",
                    "readout": "DeBERTa schema classifier", "distribution": "winning label confidence only",
                    "dtype": "float32"}

    def __init__(self, path):
        from gliner2 import AutoExtractor
        self.model = AutoExtractor.from_pretrained(path, map_location="cuda").eval()
        self.tokenizer = self.model.processor.tokenizer

    def schema(self, record):
        tasks = {}
        for key, q in record["questions"].items():
            criteria = q["criteria"]
            labels = criteria if isinstance(criteria, dict) else {str(i): str(t) for i, t in enumerate(criteria)}
            tasks[key] = {"labels": labels, "prompt": q["instructions"]}
        return tasks

    def key(self, record):
        return json.dumps(self.schema(record), ensure_ascii=False)

    def length(self, record):
        return len(self.tokenizer.encode(record["state"])) + len(json.dumps(self.schema(record)))

    def single(self, record):
        raw = self.model.classify_text(record["state"], self.schema(record), include_confidence=True)
        return {"answers": raw}

    def batch(self, records):
        if any(self.key(r) != self.key(records[0]) for r in records):
            raise ValueError("Released GLiNER batch API requires a common ordered schema")
        return [{"answers": raw} for raw in self.model.batch_classify_text(
            [r["state"] for r in records], self.schema(records[0]), batch_size=len(records), include_confidence=True)]


class D3Contract:
    capabilities = {"batch": True, "batch_key": "any text schema", "readout": "255-code head",
                    "distribution": "native categorical/ordinal probabilities; entropy confidence",
                    "dtype": "bfloat16 backbone; released readout dtype"}

    def __init__(self, path):
        sys.path.insert(0, path)
        from d3_runtime import D3
        self.runtime = D3.from_pretrained(path, device="cuda:0", batch_size=8, verify="fast")
        self.model = self.runtime.backbone
        self.tokenizer = self.runtime.tokenizer

    def key(self, record):
        return "text"

    def length(self, record):
        p = self.runtime.prepare(record["state"], record["questions"])
        return sum(len(s) for s in p.sequences.values())

    def single(self, record):
        p = self.runtime.prepare(record["state"], record["questions"])
        probs, tokens = self.runtime.run(p)
        return self.runtime.respond(p, probs, tokens)

    def batch(self, records):
        prepared = [self.runtime.prepare(r["state"], r["questions"]) for r in records]
        if any(p.errors or p.reversed_sequences for p in prepared):
            raise ValueError("Batch contract covers valid text without permutation averaging")
        rows = [(i, k, p.sequences[k], len(p.questions[k].keys))
                for i, p in enumerate(prepared) for k in p.runnable]
        distributions = [{} for _ in records]
        for offset in range(0, len(rows), self.runtime.batch_size):
            chunk = rows[offset:offset+self.runtime.batch_size]
            values = self.runtime.probabilities([r[2] for r in chunk], [r[3] for r in chunk])
            for (i, key, _, _), probabilities in zip(chunk, values):
                distributions[i][key] = probabilities
        return [self.runtime.respond(p, distributions[i], sum(len(s) for s in p.sequences.values()))
                for i, p in enumerate(prepared)]


class JevContract:
    capabilities = {"batch": True, "batch_key": "native vLLM per-request sampling parameters",
                    "readout": "decision LoRA answer-token slots with bias and calibration",
                    "distribution": "per-option probabilities; native System 1 only", "dtype": "bfloat16"}

    def __init__(self, path):
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        self.path = Path(path)
        self.head = json.loads((self.path / "adapter_vllm/decision_head.json").read_text())
        self.temperatures = json.loads((self.path / "calibration.json").read_text())["per_kind"]
        self.llm = LLM(path, enable_lora=True, max_lora_rank=32, logprobs_mode="processed_logprobs",
                       max_model_len=4096, gpu_memory_utilization=0.80, max_num_seqs=32,
                       enable_prefix_caching=True, dtype="bfloat16")
        self.lora = LoRARequest("jev-decision", 1, str(self.path / "adapter_vllm"))
        self.tokenizer = self.llm.get_tokenizer()
        self.model = None

    def key(self, record):
        return "text"

    def prepare_question(self, record, q):
        from vllm import SamplingParams
        kind = q["type"]
        if kind == "choice":
            if len(q["criteria"]) > 16:
                raise ValueError("This native-slot contract covers at most 16 choice options")
            options = [f"{k}: {v}" for k, v in q["criteria"].items()]
            lines = [f"{'ABCDEFGHIJKLMNOP'[i]}) {o}" for i, o in enumerate(options)]
        elif kind == "noul":
            options = lines = ["false", "true"]
        elif kind == "score" and q["criteria"] == [str(i) for i in range(6)]:
            options = lines = [str(i) for i in range(6)]
        else:
            raise ValueError("Native JEV score contract uses the fixed 0-5 scale")
        start = self.head["slots"]["ranges"][kind][0]
        ids = self.head["verbalizer_ids"][start:start+len(options)]
        prompt = f"[kind] {kind}\n[state] {record['state']}\n[question] {q['instructions']}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) > 4096:
            raise ValueError("Prompt exceeds the configured native context")
        params = SamplingParams(max_tokens=1, temperature=1.0, top_k=0, top_p=1.0,
                                allowed_token_ids=ids, logprobs=len(ids))
        return {"prompt_token_ids": tokens}, params, (kind, options, ids, start)

    def length(self, record):
        return sum(len(self.prepare_question(record, q)[0]["prompt_token_ids"]) for q in record["questions"].values())

    def single(self, record):
        return self.batch([record])[0]

    def batch(self, records):
        rows = [(i, key, self.prepare_question(r, q)) for i, r in enumerate(records) for key, q in r["questions"].items()]
        values = self.llm.generate([row[2][0] for row in rows], [row[2][1] for row in rows],
                                   lora_request=self.lora, use_tqdm=False)
        result = [{"answers": {}} for _ in records]
        for (i, key, (_, _, (kind, names, ids, start))), value in zip(rows, values):
            logprobs = value.outputs[0].logprobs[0]
            if any(t not in logprobs for t in ids):
                raise RuntimeError("Native vLLM did not return every requested answer token")
            logits = [(logprobs[t].logprob + self.head["bias"][start+j])/self.temperatures[kind]
                      for j, t in enumerate(ids)]
            maximum = max(logits); weights = [math.exp(z-maximum) for z in logits]; total = sum(weights)
            probs = [w/total for w in weights]; best = max(range(len(probs)), key=probs.__getitem__)
            answer = {"kind": kind, "options": names, "probabilities": probs,
                      "choice_index": best, "choice": names[best]}
            result[i]["answers"][key] = answer
        return result


def load_contract(alias):
    spec = contract_model(alias)
    return {"omni": OmniContract, "gliner": GlinerContract, "d3": D3Contract, "jev": JevContract}[spec.adapter](prepared_path(alias))
