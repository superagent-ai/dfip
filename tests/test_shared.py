from dfip.shared import BatchLimits, SharedBatcher, native_batches


class Native:
    def key(self, row):
        return row['schema']

    def length(self, row):
        return row['length']

    def single(self, row):
        return row['id']

    def batch(self, rows):
        assert len({self.key(row) for row in rows}) == 1
        return [row['id'] for row in rows]


def rows():
    return [dict(id=i, schema=key, length=n) for i, (key, n) in enumerate(
        [('a', 10), ('b', 10), ('a', 1000), ('a', 12), ('a', 11), ('b', 11)])]


def test_preserve_order_without_mixing_schemas_or_extreme_padding():
    native = Native(); data = rows()
    scheduler = SharedBatcher(native, {'a': [2, 3, 8], 'b': [2, 8]}, BatchLimits(padded_token_budget=100))
    outputs, batches = scheduler.run(data)
    assert outputs == list(range(6))
    assert batches[0] == [0, 4, 3]
    assert batches[1] == [1, 5]
    assert batches[2] == [2]


def test_unadmitted_contract_uses_native_single():
    outputs, batches = SharedBatcher(Native(), {}).run(rows())
    assert outputs == list(range(6))
    assert all(len(batch) == 1 for batch in batches)


def test_cap_and_token_budget_are_enforced():
    data = [dict(id=i, schema='a', length=30) for i in range(9)]
    _, batches = SharedBatcher(Native(), {'a': [3, 8]}, BatchLimits(padded_token_budget=100)).run(data)
    assert [len(batch) for batch in batches] == [3, 3, 3]
    _, batches = SharedBatcher(Native(), {'a': [2]}).run(data)
    assert [len(batch) for batch in batches] == [2, 2, 2, 2, 1]


def test_native_comparator_already_batches_compatible_requests():
    outputs, batches = native_batches(Native(), rows())
    assert outputs == list(range(6))
    assert batches == [[0, 2, 3, 4], [1, 5]]


def test_no_answer_cache_on_changed_inputs():
    data = rows(); scheduler = SharedBatcher(Native(), {'a': [2, 4, 8]})
    scheduler.run(data)
    data[0]['id'] = 100
    assert scheduler.run(data)[0][0] == 100


def test_partial_queue_cannot_use_a_rejected_batch_size():
    data = [dict(id=i, schema='a', length=10) for i in range(6)]
    _, batches = SharedBatcher(Native(), {'a': [4, 8]}).run(data)
    assert [len(b) for b in batches] == [4, 1, 1]
