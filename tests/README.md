# Test suites

Tests are organized by resources and boundaries:

- unit/: pure logic and isolated fakes; no CUDA required.
- integration/: CPU interactions among components, files, HTTP, threads, or processes.
- gpu/: CUDA tensors, native kernels, CUDA Graphs, or NCCL. Run manually on suitable hardware.
- support/: reusable builders and capability marks. conftest.py files contain fixtures only.

Run the full suite (GPU cases skip when requirements are unavailable):

    python -m pytest tests/ -v

Run CPU suites only when iterating locally:

    python -m pytest tests/unit tests/integration -q

Run GPU tests on a machine with the required devices and extensions:

    python -m pytest tests/gpu -q

GPU tests skip when CUDA, the required extension, or enough visible devices are unavailable. A successful command with skips does not mean the skipped paths were exercised. The move into tier directories changes pytest node ID paths; test function names remain the same where possible.
