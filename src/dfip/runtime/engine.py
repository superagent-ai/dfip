"""Thread-safe, text-only decision inference with measured inference defaults."""

import threading
from dfip.runtime.adapters import load_adapter
from dfip.runtime.optimization import optimize


class DecisionEngine:
    def __init__(self, model_id, adapter=None):
        if model_id == "superagent-ai/security-one-27b":
            raise ValueError("Use the SGLang client for Security-One")
        self.model_id = model_id
        self.adapter = adapter if adapter is not None else load_adapter(model_id)
        self.lock = threading.RLock()
        self.optimization = optimize(self.adapter, "production", [])
        self.adapter.enable_graphs(True)

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
                raise ValueError(
                    "Each record needs state and a nonempty questions mapping"
                )
            if r.get("images") or r.get("videos"):
                raise ValueError("This DFIP release covers text decisions")
        with self.lock:
            return self.adapter.predict(records)

    def diagnostics(self):
        with self.lock:
            return {
                "model_id": self.model_id,
                "optimization": self.optimization,
                "graphs": self.adapter.graph_stats(),
                "current_candidate_tokens": len(self.adapter.model.lm_head.ids)
                if hasattr(self.adapter.model, "lm_head")
                and hasattr(self.adapter.model.lm_head, "ids")
                else None,
            }
