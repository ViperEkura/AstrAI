# Tests

Tests are organized by module: config, data, extension, inference, module,
optim, parallel, trainer, and the other existing components. Each behavior has
one test body; CPU and CUDA variants share parameters where their contracts
are identical.

The shared cpu_device fixture is an explicit CPU baseline. CUDA-only model
kernel tests request cuda_device; a directory-wide device override must not
change the execution device of unrelated tests.

Reusable builders remain in tests/support. Domain conftest.py files own fixtures.
There are no empty package initializers; pytest uses importlib import mode so
same-named files in different modules do not collide. Component factories and
spawned workers remain importable by their full module paths.

## Running tests

Install the existing development extras:

```bash
pip install -e '.[dev]'
```

Run all tests, including available CUDA cases (CI and the pre-commit script use this):

```bash
python -m pytest tests -q
```

Run CPU tests without scheduling CUDA cases when other jobs occupy the GPUs:

```bash
python -m pytest tests -m "not cuda" -q
```

Run CUDA tests on hardware with the required devices and extensions:

```bash
python -m pytest tests -m cuda -q
```

The distributed marker selects multi-rank/worker scenarios independently of
device type. CPU/Gloo scenarios remain part of the CPU suite. The slow marker
identifies long-running tests. Unknown markers are errors.

## Skips and numerical checks

- Skip only for unavailable devices, compiled extensions, or explicitly required
  environment capabilities. Runtime failures under valid test prerequisites fail.
- Existing capability guards live in tests/support/capabilities.py. Resource
  markers select tests; they do not replace device-count or kernel checks.
- Tokenizer tests use the existing runtime dependencies and explicit expected
  values. The existing Qwen3 interoperability check is optional: it runs when
  transformers>=4.51.3 is already installed, otherwise reports a skip.
- Preserve independent forward/gradient references, accumulation equivalence,
  checkpoint continuation, and failure/cleanup assertions. Numerical tolerances
  are explicit for each algorithm and dtype.
- Inspect skips with -ra. A passing CPU run does not validate CUDA/NCCL kernels.

## Inventory and checks

```bash
python -m pytest tests --collect-only -q
python -m pytest tests -m cuda --collect-only -q
bash scripts/pre_commit.sh --skip-deps
```

scripts/pre_commit.sh runs formatting, import sorting, and the full test suite.
The existing GitHub lint and full-test workflows remain separate.

## Helper names

Functions shared across test modules use public names without a leading
underscore, including component/string entrypoints. Functions used only inside
their defining module keep a leading underscore. Shared builders live in the
existing support modules; do not add forwarding aliases for old names.

Testing a production private method does not make it a public production API.
Use fixtures for resource ownership and teardown, and ordinary functions for
data/model construction.

## Compression priorities

Judge necessity by failure severity, unique regression coverage, an independent
reference, tolerance of internal refactoring, and execution cost. Prefer public
behavior checks; inspect internals for gradients, masks, state ownership,
resource release, and explicit architecture/performance contracts.

| Content | Treatment |
| --- | --- |
| Numerical references, gradients, accumulation, continuation and resource isolation | Preserve independent references, failure boundaries and dtype/algorithm tolerances; share repeated preparation. |
| Functional modes and transformations | Parameterize the owning module's cases; upper layers cover composition and real I/O. |
| Simple assignments, forwarding, registration, constant defaults and constructor field readback | Remove standalone checks and their dedicated doubles, even without equivalent replacement. |
| Weak shape/nonempty smoke, private implementation mirrors and third-party behavior | Remove where they add little protection; retain real ABI, memory and lifecycle boundaries. |
| Dead helpers and single-consumer wrappers | Delete or move ordinary helpers to their consumer; keep spawned factories importable. |

Test counts may decrease when low-risk coverage is deliberately removed. Record
those groups in the PR, separately from merged coverage and protected regressions.
Do not relax numerical tolerances, move test code elsewhere, or compress layout
to meet a line quota. Remove section banners and comments that repeat test names;
retain formulas, tolerance rationale and regression/failure explanations.
