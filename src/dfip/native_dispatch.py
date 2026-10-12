"""Request dispatch policy; experimental head mutations remain serialized."""

from contextlib import nullcontext
from threading import RLock


def dispatch_lock(head_mode, concurrent=False):
    if concurrent and head_mode != 'native':
        raise ValueError('Concurrent dispatch requires the unmodified native SGLang head')
    return nullcontext() if concurrent else RLock()
