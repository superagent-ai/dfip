import collections
import dataclasses
import types

import torch
from dfip.runtime.kernels import candidate_projection, rms_norm


def fingerprint(obj):
    if isinstance(obj, torch.Tensor):
        return (
            "tensor",
            tuple(obj.shape),
            tuple(obj.stride()),
            str(obj.dtype),
            str(obj.device),
        )
    if isinstance(obj, dict):
        return tuple((k, fingerprint(v)) for k, v in obj.items())
    if isinstance(obj, (tuple, list)):
        return tuple(fingerprint(v) for v in obj)
    if dataclasses.is_dataclass(obj):
        return tuple(
            (f.name, fingerprint(getattr(obj, f.name))) for f in dataclasses.fields(obj)
        )
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    raise TypeError(f"Unsupported graph metadata: {type(obj)}")


def clone_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: clone_tree(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(clone_tree(v) for v in obj)
    if isinstance(obj, list):
        return [clone_tree(v) for v in obj]
    if dataclasses.is_dataclass(obj):
        return dataclasses.replace(
            obj,
            **{
                f.name: clone_tree(getattr(obj, f.name))
                for f in dataclasses.fields(obj)
            },
        )
    return obj


def copy_tree(dst, src):
    if isinstance(src, torch.Tensor):
        dst.copy_(src)
        return
    if isinstance(src, dict):
        for k in src:
            copy_tree(dst[k], src[k])
    elif isinstance(src, (tuple, list)):
        for d, s in zip(dst, src):
            copy_tree(d, s)
    elif dataclasses.is_dataclass(src):
        for f in dataclasses.fields(src):
            copy_tree(getattr(dst, f.name), getattr(src, f.name))


class GraphRunner:
    """Bounded shape cache. Captures computation, never caches predictions."""

    def __init__(self, fn, max_entries=4):
        self.fn = fn
        self.max_entries = max_entries
        self.cache = collections.OrderedDict()
        self.failures = collections.OrderedDict()
        self.captures = 0
        self.replays = 0
        self.capture_min_hits = 1
        self.shape_hits = collections.OrderedDict()
        self.enabled = True
        self.sequence_limit = None
        self.batch_limit = None
        self.include_padding_values = False

    def __call__(self, *args, **kwargs):
        if not self.enabled:
            return self.fn(*args, **kwargs)
        primary = kwargs.get("input_ids")
        if primary is None:
            primary = kwargs.get("inputs_embeds")
        if isinstance(primary, torch.Tensor) and primary.ndim >= 2:
            if (
                self.sequence_limit is not None
                and primary.shape[1] > self.sequence_limit
            ):
                return self.fn(*args, **kwargs)
            if self.batch_limit is not None and primary.shape[0] > self.batch_limit:
                return self.fn(*args, **kwargs)
        try:
            key = fingerprint((args, kwargs))
        except TypeError:
            return self.fn(*args, **kwargs)
        if self.include_padding_values:
            padding = kwargs.get("attention_mask")
            if isinstance(padding, dict):
                padding = padding.get("full_attention")
            if isinstance(padding, torch.Tensor):
                # Hybrid recurrence launch metadata depends on the valid lengths,
                # even when the padded tensor shape is unchanged.
                lengths = (
                    padding.bool().reshape(padding.shape[0], -1).sum(-1).cpu().tolist()
                )
                key = (key, ("valid_lengths", tuple(lengths)))
        if key in self.failures:
            return self.fn(*args, **kwargs)
        if key not in self.cache and self.capture_min_hits > 1:
            self.shape_hits[key] = self.shape_hits.get(key, 0) + 1
            self.shape_hits.move_to_end(key)
            if len(self.shape_hits) > 64:
                self.shape_hits.popitem(last=False)
            if self.shape_hits[key] < self.capture_min_hits:
                return self.fn(*args, **kwargs)
        if key not in self.cache:
            static_args, static_kwargs = clone_tree((args, kwargs))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.fn(*static_args, **static_kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = self.fn(*static_args, **static_kwargs)
            except Exception:
                import traceback

                self.failures[key] = traceback.format_exc()[-6000:]
                if len(self.failures) > 32:
                    self.failures.popitem(last=False)
                return self.fn(*args, **kwargs)
            self.cache[key] = (static_args, static_kwargs, graph, out)
            self.captures += 1
            if len(self.cache) > self.max_entries:
                self.cache.popitem(last=False)
        static_args, static_kwargs, graph, out = self.cache[key]
        self.cache.move_to_end(key)
        copy_tree((static_args, static_kwargs), (args, kwargs))
        graph.replay()
        self.replays += 1
        return out


class CandidateHead(torch.nn.Module):
    """Scores a verified union of answer tokens, retaining the native readout API."""

    def __init__(self, head, token_ids, use_triton):
        super().__init__()
        self.weight = head.weight
        ids = torch.tensor(
            sorted(set(token_ids)), device=head.weight.device, dtype=torch.long
        )
        self.register_buffer("ids", ids)
        self.register_buffer(
            "selected", head.weight.detach().index_select(0, ids).contiguous()
        )
        self.use_triton = use_triton
        self.coverage = set(map(int, token_ids))

    def ensure_tokens(self, token_ids):
        extra = set(map(int, token_ids)) - self.coverage
        if not extra:
            return False
        self.coverage.update(extra)
        self.ids = torch.tensor(
            sorted(self.coverage), device=self.weight.device, dtype=torch.long
        )
        self.selected = self.weight.detach().index_select(0, self.ids).contiguous()
        return True

    def forward(self, x):
        values = (
            candidate_projection(x, self.selected)
            if self.use_triton
            else torch.nn.functional.linear(x, self.selected)
        )
        full = torch.full(
            (*x.shape[:-1], self.weight.shape[0]),
            -torch.inf,
            device=x.device,
            dtype=values.dtype,
        )
        return full.index_copy(-1, self.ids, values)


def install_norms(model, allowed_sources):
    installed = []
    for name, m in model.named_modules():
        typename = type(m).__name__
        if typename not in allowed_sources:
            continue
        spec = allowed_sources[typename]
        eps = getattr(m, "eps", getattr(m, "variance_epsilon", 1e-6))

        def forward(self, x, _eps=eps, _spec=spec):
            return rms_norm(x, self.weight, _eps, **_spec)

        m.forward = types.MethodType(forward, m)
        installed.append(name)
    return installed
