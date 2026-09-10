# Inference runtime release/resume on NVIDIA L20

Implementation commit: `5900c786322b522162af0bef0674464806f1628a`

Environment: NVIDIA L20, PyTorch 2.11.0+cu128, CUDA 12.8, BF16. The
`astrai-1b` preset used 24 layers, hidden size 1536, four KV heads, batch size
4, prompt length 128, and four greedy decode tokens. CUDA graphs were disabled
to isolate the scheduler-owned KV and workspace lifecycle. Each context bound
was measured across five complete release/resume/output-parity cycles.

| Max context | Runtime footprint | Reclaimed | Reclaimed % | Release median | Resume median | Greedy parity |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 200.97 MiB | 192.85 MiB | 95.96% | 105.62 ms | 2.01 ms | 5/5 |
| 8,192 | 777.14 MiB | 769.01 MiB | 98.95% | 90.16 ms | 5.09 ms | 5/5 |
| 32,768 | 3,081.81 MiB | 3,073.68 MiB | 99.74% | 89.27 ms | 5.10 ms | 5/5 |

`release()` includes scheduler stop, Python reference collection, and CUDA
allocator cache release. `resume()` reconstructs the cache and executor; the
reported resume latency excludes the first generation after reconstruction.
The model-only allocation remained resident throughout every cycle.

Reproduce one cell with:

```bash
python scripts/tools/benchmark_inference_lifecycle.py \
  --preset astrai-1b \
  --batch-size 4 \
  --max-seq-len 32768 \
  --prompt-len 128 \
  --max-tokens 4 \
  --trials 5 \
  --no-cuda-graph
```

Raw results:

- `inference_release_resume_l20_2048.json`
- `inference_release_resume_l20_8192.json`
- `inference_release_resume_l20_32768.json`

## 100-cycle stability soak

A separate three-GPU soak ran the tiny deterministic preset concurrently on
three L20s so that every context bound completed 100 release/resume/output
parity cycles. This is lifecycle-stability evidence; its memory footprint is
not comparable to the `astrai-1b` table above.

| Max context | Cycles | Reclaimed | Reclaimed % | Release median / p99 | Resume median / p99 | Greedy parity |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 100 | 0.572 MiB | 6.58% | 72.201 / 88.869 ms | 0.410 / 0.553 ms | 100/100 |
| 8,192 | 100 | 2.239 MiB | 21.61% | 74.530 / 100.495 ms | 0.581 / 6.293 ms | 100/100 |
| 32,768 | 100 | 8.907 MiB | 52.30% | 68.257 / 93.089 ms | 0.441 / 0.557 ms | 100/100 |

The summarized machine-readable result is
`inference_release_resume_l20_100_cycles.json`.
