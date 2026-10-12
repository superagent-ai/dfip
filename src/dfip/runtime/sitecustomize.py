"""Opt-in SGLang decision-only readout; never enable for general generation."""

import os

if os.environ.get("DFIP_SGLANG_HEAD") in ("cublas", "triton"):
    import json
    import torch
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from dfip.runtime.kernels import candidate_projection

    original = LogitsProcessor._compute_lm_head
    token_ids = json.loads(os.environ["DFIP_SGLANG_TOKEN_IDS"])
    if not token_ids or len(set(token_ids)) != len(token_ids):
        raise ValueError("DFIP requires a unique, verified answer-token union")

    def decision_head(self, hidden_states, lm_head, embedding_bias=None):
        if embedding_bias is not None or not hasattr(lm_head, "weight"):
            raise ValueError(
                "Unsupported head: decision specialization requires an unbiased BF16 head"
            )
        weight = lm_head.weight
        if weight.dtype != torch.bfloat16 or getattr(lm_head, "tp_size", 1) != 1:
            raise ValueError("DFIP SGLang supports a single GPU BF16 head")
        if not hasattr(lm_head, "_dfip_selected"):
            ids = torch.tensor(token_ids, device=weight.device, dtype=torch.long)
            if min(token_ids) < 0 or max(token_ids) >= weight.shape[0]:
                raise ValueError("Invalid answer token")
            lm_head._dfip_selected = weight.index_select(0, ids).contiguous()
            lm_head._dfip_ids = ids
            print(
                json.dumps(
                    {
                        "dfip_decision_head": True,
                        "selected_rows": len(ids),
                        "vocabulary_rows": weight.shape[0],
                        "mode": os.environ["DFIP_SGLANG_HEAD"],
                        "fp32_lm_head": self.use_fp32_lm_head,
                    }
                ),
                flush=True,
            )
        x = hidden_states.to(weight.dtype)
        selected = lm_head._dfip_selected
        if os.environ["DFIP_SGLANG_HEAD"] == "triton" and not self.use_fp32_lm_head:
            values = candidate_projection(x, selected)
        elif self.use_fp32_lm_head:
            values = torch.mm(x, selected.T, out_dtype=torch.float32)
        else:
            values = torch.mm(x, selected.T)
        # Conditional probability readout cancels the full-vocabulary normalizer.
        # The unused generated token is discarded by the decision client.
        logits = torch.zeros(
            (x.shape[0], weight.shape[0]), device=x.device, dtype=values.dtype
        )
        return logits.index_copy(1, lm_head._dfip_ids, values)

    LogitsProcessor._compute_lm_head = decision_head
    print(
        json.dumps(
            {"dfip_patch_installed": True, "mode": os.environ["DFIP_SGLANG_HEAD"]}
        ),
        flush=True,
    )
