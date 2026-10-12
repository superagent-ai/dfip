"""Record the implementation actually bound by Transformers' fallback decorator."""

import inspect


def callable_provenance(function):
    values=inspect.getclosurevars(function).nonlocals if inspect.isfunction(function) else {}
    implementation=values.get('implementation',function)
    return {'wrapper':function.__module__+'.'+function.__qualname__,
            'implementation':implementation.__module__+'.'+implementation.__qualname__}


def qwen_backend():
    import transformers.models.qwen3_5.modeling_qwen3_5 as q
    return {'causal_conv1d':callable_provenance(q.causal_conv1d_fn),
            'chunk_gated_delta_rule':callable_provenance(q.torch_chunk_gated_delta_rule),
            'recurrent_gated_delta_rule':callable_provenance(q.torch_recurrent_gated_delta_rule)}
