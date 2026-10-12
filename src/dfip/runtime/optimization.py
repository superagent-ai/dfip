"""Model adapters for a decision-focused inference path."""

import importlib
import types

import torch

from dfip.runtime.optimizations import CandidateHead, GraphRunner, install_norms


def attach_graph(adapter, owner, name):
    fn = getattr(owner, name)
    if isinstance(fn, GraphRunner):
        fn = fn.fn
    graph = GraphRunner(fn)
    setattr(owner, name, graph)
    adapter.graphs.append(graph)


def optimize(adapter, mode, records):
    if mode == "production":
        selected = (
            "dfip" if type(adapter).__name__ == "LiquidAdapter" else "native_graphs"
        )
        report = optimize(adapter, selected, records)
        report["mode"] = "production"
        for graph in adapter.graphs:
            graph.capture_min_hits = 3
        report["graph_admission_hits"] = 3
        if type(adapter).__name__ in ("LiquidAdapter", "ClefAdapter", "DeciderAdapter"):
            for graph in adapter.graphs:
                graph.sequence_limit = 512
            report["graph_sequence_limit"] = 512
        if type(adapter).__name__ in ("ClefAdapter", "DeciderAdapter"):
            for graph in adapter.graphs:
                graph.batch_limit = 1
            report["graph_batch_limit"] = 1
            if type(adapter).__name__ == "DeciderAdapter":
                report["algorithm_changes"].append(
                    "Native fallback for multi-question/batched forwards; hybrid padded graph stress test failed"
                )
        return report
    if hasattr(adapter, "_native_predict"):
        adapter.predict = adapter._native_predict
    name = type(adapter).__name__
    adapter.graphs = []
    report = {"mode": mode, "kernels": [], "algorithm_changes": []}
    if mode == "native_compile":
        if name == "LiquidAdapter":
            adapter.engine._one_pass = torch.compile(
                adapter.model.forward, mode="reduce-overhead"
            )
        elif name == "JuliaAdapter":
            adapter.engine.forward = torch.compile(adapter.model, dynamic=True)
        else:
            raise ValueError("Compiler baseline currently covers Julia and Liquid")
        adapter.enable_graphs = lambda enabled: None
        adapter.graph_stats = lambda: []
        report["algorithm_changes"].append(
            "Released native torch.compile option; no custom kernels or candidate pruning"
        )
        return report

    if mode in ("dfip", "readout_only", "norm_only"):
        if name == "JuliaAdapter":
            adapter.model.marker_only_head = True
            report["algorithm_changes"].append(
                "Exact final decision-head queries restricted to option positions; upstream native feature enabled on GPU"
            )
        elif name == "LiquidAdapter":
            prompt = importlib.import_module(
                type(adapter.engine).__module__.rsplit(".", 1)[0] + ".prompt"
            )
            ids = []
            for r in records:
                for q in r["questions"].values():
                    ids.extend(
                        t
                        for group in prompt.readout_ids(
                            adapter.engine.tokenizer, prompt.as_question(q)
                        )
                        for t in group
                    )
            if mode != "norm_only":
                head = CandidateHead(adapter.model.lm_head, ids, use_triton=True)
                adapter.model.lm_head = head
                report["algorithm_changes"].append(
                    "Project only verified answer-token union; preserve native max-alias pooling and candidate normalization"
                )
                report["candidate_tokens"] = len(head.ids)
                adapter._native_predict = adapter.predict

                def guarded_predict(batch):
                    current = [
                        t
                        for r in batch
                        for q in r["questions"].values()
                        for group in prompt.readout_ids(
                            adapter.engine.tokenizer, prompt.as_question(q)
                        )
                        for t in group
                    ]
                    if head.ensure_tokens(current):
                        for graph in adapter.graphs:
                            graph.cache.clear()
                            graph.failures.clear()
                    return adapter._native_predict(batch)

                adapter.predict = guarded_predict
                report["kernels"].append("Triton candidate projection")
            if mode == "norm_only":
                report["norm_modules"] = install_norms(
                    adapter.model, {"RMSNorm": {"round_before_weight": True}}
                )
                report["kernels"].append(
                    "Triton RMSNorm preserving intermediate BF16 rounding"
                )
        elif name in ("ClefAdapter", "DeciderAdapter", "SecurityAdapter"):
            report["norm_modules"] = install_norms(
                adapter.model,
                {"Qwen3_5RMSNorm": {"offset": 1.0, "round_before_weight": False}},
            )
            if report["norm_modules"]:
                report["kernels"].append("Triton Qwen RMSNorm with offset weights")
            if name == "SecurityAdapter":
                # This checkpoint exports a full-vocabulary head; only answer-code rows are consumed.
                adapter.reduced_head = torch.nn.Linear(
                    adapter.model.config.text_config.hidden_size,
                    len(adapter.label_ids),
                    bias=False,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                adapter.reduced_head.weight = torch.nn.Parameter(
                    adapter.model.lm_head.weight[adapter.label_ids]
                    .detach()
                    .contiguous(),
                    requires_grad=False,
                )
                adapter.decision_only = True
                report["algorithm_changes"].append(
                    "Exact answer-code readout directly from final hidden state"
                )
    if name == "JuliaAdapter":
        # Preserve ModernBERT masks, creating position indices on the GPU so capture is legal.
        def update_mask(self, attention_mask, output_attentions=False):
            from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

            full = _prepare_4d_attention_mask(attention_mask, self.dtype)
            positions = torch.arange(full.shape[2], device=attention_mask.device)
            window = (
                positions[:, None] - positions[None, :]
            ).abs() <= self.config.local_attention // 2
            return full, full.masked_fill(
                ~window[None, None], torch.finfo(self.dtype).min
            )

        adapter.model.encoder._update_attention_mask = types.MethodType(
            update_mask, adapter.model.encoder
        )
        report["algorithm_changes"].append(
            "Equivalent ModernBERT local mask built on GPU for graph capture"
        )
        attach_graph(adapter, adapter.engine, "forward")
    elif name == "LiquidAdapter":
        attach_graph(adapter, adapter.engine, "_one_pass")
    elif name == "ClefAdapter":
        attach_graph(
            adapter, adapter.model.language_model.model.language_model, "forward"
        )
    elif name == "DeciderAdapter":
        attach_graph(adapter, adapter.model.backbone.language_model, "forward")
        for graph in adapter.graphs:
            graph.include_padding_values = True
        report["algorithm_changes"].append(
            "Graph key includes per-row valid lengths for hybrid recurrence metadata"
        )
    elif name == "SecurityAdapter":
        attach_graph(adapter, adapter.model.model.language_model, "forward")

    def enable_graphs(enabled):
        for graph in adapter.graphs:
            graph.enabled = enabled

    def graph_stats():
        return [
            {
                "captures": g.captures,
                "replays": g.replays,
                "fallback_shapes": len(g.failures),
                "fallback_reasons": list(g.failures.values())[:4],
            }
            for g in adapter.graphs
        ]

    adapter.enable_graphs = enable_graphs
    adapter.graph_stats = graph_stats
    enable_graphs(False)
    return report
