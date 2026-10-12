# Contributing

DFIP separates the lightweight client (`src/dfip/client.py`) from GPU code
(`src/dfip/runtime/`) and Modal deployment (`src/dfip/cloud/`). Cloud images contain
the package source, so wheel and editable installs both deploy from any working
directory. GPU libraries should never be imported by the top-level package.

To add a model:

1. Pin an immutable checkpoint revision in `models.py` and review its license.
2. Implement an adapter that preserves the released prompt, normalization,
   calibration, pooling, alias handling, and typed outputs.
3. Establish a native baseline before adding acceleration. Use native fallbacks
   for unvalidated shapes, batching, dtypes, or candidate schemas.
4. Add a Modal class/image and client routing. Keep secrets only in preparation
   unless the serving process actually needs them.
5. Run paired benchmarks, changed-value graph replay, dynamic candidate coverage,
   and full model fidelity checks on Modal. Report probability drift and argmax
   changes as well as latency, cold setup, memory, and workload limits.

Do not promote an optimization on microkernel timing alone. Preserve raw paired
samples and document benchmark overlap, hardware, exact model/image revisions,
and whether network and cold starts are included. The public typed-decision
corpus does not replace a domain-specific security evaluation.

Run `pytest` and `ruff check src tests` before submitting a pull request. Never
commit model weights, datasets, credentials, private request data, or cloud logs
containing them. Examples must contain synthetic or public data.
