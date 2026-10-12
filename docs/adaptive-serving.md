# Experimental support for varied traffic

The native app remains the default. The review branch adds two ways to handle changing requests:

- Security-One dispatches independent requests concurrently to its native SGLang scheduler. Its inference head remains native. Three v3 rounds retained all 5,400 requests offered at 30 requests/s, with 29.76 within-window completions/s and geometric-mean successful-request p95 0.65 s. This is minute-scale evidence for this configuration, not a capacity guarantee for other workloads.
- Clef and Decider have a separate opt-in app with bounded graph admission and native fallback. It avoids repeated graph eviction and recapture. The v4 pilot is exploratory; the original CUDA failure is not localized or proven fixed.

## Use the native app

Deploy the normal app following the main README. Security-One's review source uses concurrent native dispatch; no existing deployed app was changed by this study. Submit requests concurrently through `Client.predict_async` to allow the native scheduler to overlap them. A sequence of awaited calls remains serial.

## Opt in to adaptive replay

Prepare Clef or Decider in your own Modal volume using `dfip prepare`, as described in the main README. Deploy the experimental app under a distinct name:

```sh
export DFIP_APP=my-dfip-adaptive
export DFIP_VOLUME=my-dfip-models
modal deploy -m dfip.cloud.serve_adaptive
```

Then use the ordinary client and answer schema:

```python
from dfip import Client

client = Client('clef', app_name='my-dfip-adaptive')
records = [{
    'id': 'ticket-1',
    'state': 'I was charged twice. Please refund the duplicate payment.',
    'questions': {'team': {
        'type': 'choice',
        'instructions': 'Which team should handle this request?',
        'criteria': {'billing': 'Payments and refunds', 'technical': 'Software faults'},
    }},
}]
answers = client.predict(records)
print(answers)
print(client.diagnostics())
```

Choose `Client('decider', app_name='my-dfip-adaptive')` for Decider. The experimental app has no other model classes. Each class has a one-container cap; invoking both model classes can still allocate two GPUs, so use one model at a time if you need one GPU across the app. The benchmark driver enforces this global sequencing separately.

The API warms 128 native diagnostic requests before enabling adaptive admission. Graph replay is eligible only for a single categorical question on one record; Boolean, ordinal, batched and multi-question API calls execute natively. The inherited capture limits are one forward row and at most 512 tokens. Prompt encoding, labels, answer conversion and checkpoint weights use the released model implementation.

Admission observes up to 128 eligible calls. Admission waits for 32 eligible calls in the window; a shape then needs at least eight matching hits and 12.5% frequency. Each engine permits at most four capture attempts, including failed attempts. Captured entries are pinned: there is no eviction or automatic replacement. A later hot shape falls back to native inference once the budget is exhausted. A new worker initializes adaptive mode and starts a new budget. The cache stores computation, never previous answers.

GPU execution remains serialized because replay buffers are mutable. This policy bounds admission work; it does not create dynamic batching or increase native capacity for every workload. Under overload, a caller still needs its own queue and timeout policy. Diagnostics expose captures, replays, resident entries, admission attempts, and native fallback counts.

Clef's validated adaptive pilot still changed returned confidence by up to 0.0065 and crossed the 0.9 threshold on fourteen matched pairs, despite preserving all 3,844 compared categorical choices. Review the separate [pilot results](../benchmarks/adaptive-study.md) and your application's numerical tolerance before using the opt-in path. An illegal-memory-access fault is propagated, not treated as a recoverable native fallback in a damaged CUDA context.
