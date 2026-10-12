"""Experimental contract-guided batching without model-name optimization rules.

Adapters own native semantics. Admission is finite empirical evidence, not a
guarantee of equivalence on future inputs. No weights or answers are cached.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchLimits:
    max_batch: int = 8
    padded_token_budget: int = 8192
    max_length_ratio: float = 4.0
    wait_ms: float = 2.0


class SharedBatcher:
    def __init__(self, adapter, admitted, limits=BatchLimits()):
        self.adapter = adapter
        self.admitted = dict(admitted)
        self.limits = limits

    def select(self, pending, lengths):
        """Return positions, always including the oldest request.

        Callers pass only arrived requests. Similar-length compatible requests
        are preferred; a fixed budget bounds padding and batch size. Ineligible
        requests run through the released single-request path.
        """
        if not pending:
            return []
        key = self.adapter.key(pending[0])
        cap = min(self.limits.max_batch, self.admitted.get(key, 1))
        if cap <= 1:
            return [0]
        selected = [0]
        candidates = sorted(range(1, len(pending)), key=lambda i: (abs(lengths[i]-lengths[0]), i))
        low = high = max(1, lengths[0])
        for i in candidates:
            if len(selected) >= cap:
                break
            if self.adapter.key(pending[i]) != key:
                continue
            new_low, new_high = min(low, max(1, lengths[i])), max(high, max(1, lengths[i]))
            if new_high/new_low > self.limits.max_length_ratio:
                continue
            if new_high*(len(selected)+1) > self.limits.padded_token_budget:
                continue
            selected.append(i); low, high = new_low, new_high
        return selected

    def execute(self, rows):
        if len(rows) == 1:
            return [self.adapter.single(rows[0])]
        return self.adapter.batch(rows)

    def length(self, row):
        # A rejected contract needs no length analysis to execute natively.
        return self.adapter.length(row) if self.admitted.get(self.adapter.key(row), 1) > 1 else 0

    def run(self, records, lengths=None):
        lengths = lengths or [self.length(r) for r in records]
        remaining = list(range(len(records)))
        outputs = [None]*len(records)
        batches = []
        while remaining:
            positions = self.select([records[i] for i in remaining], [lengths[i] for i in remaining])
            indices = [remaining[p] for p in positions]
            values = self.execute([records[i] for i in indices])
            if len(values) != len(indices):
                raise RuntimeError("Native batch returned the wrong number of requests")
            for i, value in zip(indices, values):
                outputs[i] = value
            batches.append(indices)
            chosen = set(positions)
            remaining = [i for p, i in enumerate(remaining) if p not in chosen]
        return outputs, batches


def native_batches(adapter, records, max_batch=8):
    """Strong native batching comparator: group by released schema constraints.

    Compatible requests retain FIFO order; output order is restored. This
    comparator gets the same native batch API and size ceiling as DFIP.
    """
    groups = {}
    for i, row in enumerate(records):
        groups.setdefault(adapter.key(row), []).append(i)
    outputs = [None]*len(records); batches = []
    for indices in groups.values():
        for offset in range(0, len(indices), max_batch):
            chunk = indices[offset:offset+max_batch]
            rows = [records[i] for i in chunk]
            values = [adapter.single(rows[0])] if len(rows) == 1 else adapter.batch(rows)
            if len(values) != len(chunk):
                raise RuntimeError("Native batch returned the wrong number of requests")
            for i, value in zip(chunk, values):
                outputs[i] = value
            batches.append(chunk)
    return outputs, batches
