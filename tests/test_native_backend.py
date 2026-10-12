from dfip.runtime.native_backend import callable_provenance


def test_decorator_name_cannot_hide_bound_fallback_implementation():
    def real_extension(x):
        return x
    implementation=real_extension
    def wrapped(x):
        return implementation(x)
    wrapped.__module__='transformers.models.qwen3_5'
    report=callable_provenance(wrapped)
    assert report['wrapper'].startswith('transformers.')
    assert report['implementation'].endswith('real_extension')
