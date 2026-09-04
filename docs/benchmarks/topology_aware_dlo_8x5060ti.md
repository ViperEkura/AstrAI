# Topology-aware DLO on 8x RTX 5060 Ti

This benchmark validates AstrAI's DP/SP rank planner with actual NCCL collectives on
the target host. The collective section models finalized one-byte INT8/FP8 DLO
weights. A separate end-to-end section validates the same DLO group semantics with
an online-FP8 MiniMax-H3 service.

## Environment

- 8x NVIDIA GeForce RTX 5060 Ti 16 GiB, compute capability 12.0
- PyTorch 2.11.0+cu128, CUDA 12.8, NCCL 2.28.9
- One NUMA node; PHB pairs `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`; other pairs `NODE`
- DLO AllGather reconstructs a 256 MiB full `uint8` payload
- SP all-to-all sends 64 MiB per rank
- 5 warmups, 20 operations per trial, 7 trials
- Each sample is the slowest rank, so one straggler cannot be hidden by averaging

Reproduce with:

```bash
python scripts/tools/benchmark_dlo_topology.py \
  --output dlo-topology.json \
  --markdown-output dlo-topology.md \
  --dp-sizes 1,2,4,8 \
  --dlo-payload-mib 256 \
  --sp-payload-mib 64 \
  --warmup 5 \
  --iterations 20 \
  --trials 7
```

## Results

| Candidate | Device order | DLO group | DLO median / p99 (ms) | SP median / p99 (ms) | Combined median (ms) |
|---|---|---|---:|---:|---:|
| DP1 x SP8 natural | `0,1,2,3,4,5,6,7` | SP8 | 41.1263 / 41.1770 | 20.7118 / 20.8178 | 61.8381 |
| DP2 x SP4 natural | `0,1,2,3,4,5,6,7` | 4x DP2 `NODE` | 57.3858 / 67.7326 | 18.4796 / 18.5259 | 75.8654 |
| DP2 x SP4 topology | `0,2,4,6,1,3,5,7` | 4x DP2 `PHB` | 52.6750 / 52.8270 | 18.8081 / 18.8933 | **71.4831** |
| DP4 x SP2 natural | `0,1,2,3,4,5,6,7` | 2x DP4 `NODE` | 43.9203 / 43.9913 | 11.3262 / 11.5051 | **55.2465** |
| DP4 x SP2 label optimum | `0,2,1,3,4,6,5,7` | 2x DP4 mixed | 44.0285 / 44.0651 | 16.5605 / 16.6675 | 60.5890 |
| DP8 x SP1 natural | `0,1,2,3,4,5,6,7` | DP8 | 41.0465 / 41.0798 | 0.3494 / 0.3497 | 41.3959 |

The raw 7-trial capture has SHA-256
`98fbb8966b7aa0a032ca1fa269ee1cd240d34b0bc4a5d918e3fdad31b040212f`;
the checked-in machine-readable summary is
[`benchmarks/results/dlo_topology_8x5060ti.json`](../../benchmarks/results/dlo_topology_8x5060ti.json).

## MiniMax-H3 online-FP8 end-to-end validation

The same host also ran current vLLM-Omni main (`e51fe6ec1`) with PyTorch
2.13.0+cu132, CUDA 13.2, and NCCL 2.29.7. MiniMax-H3 was loaded with online FP8,
DP1 x SP8, text-encoder TP8, VAE tile parallelism across eight GPUs, and distributed
layerwise offload with AllGather. Startup selected the SP8 group for DLO, allocated a
615.6 MiB maximum full FP8 block plus a 76.9 MiB shard per rank, and completed engine
initialization in 306.46 seconds. Idle process-scoped GPU memory was 1.86 GiB per
worker.

All requests used T2VA, 832x480, 24 FPS, a nominal four-second duration, and seed
1101. The model produced 107 frames and 4.45 seconds of 32 kHz audio. The one-step
probe was rejected by H3's expected minimum-two-entry sigma schedule and is excluded
from the benchmark.

| Sample | Steps | Client E2E (s) | Server E2E (s) | Denoise latency (ms/step) | Peak GPU memory (MiB) |
|---|---:|---:|---:|---:|---:|
| First valid request | 2 | 17.317 | 16.877 | 8,437.763 | 13,998 |
| Warm request | 2 | 13.714 | 13.292 | 6,645.439 | 14,174 |
| Stability request | 10 | 71.719 | 71.320 | 7,131.966 | 14,206 |

All eight GPUs reached 100% utilization during every valid sample. The 10-step
request retained 2,105 MiB of headroom on a 16,311 MiB GPU and returned a 1,022,884
byte MP4 (`1b3e2d9a13ed292c2bb16ab09fe795b9ec27e4038755ba70ca8c5d72dbc0fdfc`).
`ffprobe` verified an 832x480 H.264 stream at 24 FPS and stereo AAC at 32 kHz.

## Interpretation

For two concurrent requests, mapping each DLO DP2 group to a PHB pair reduces the
combined median by 5.78% and removes the natural mapping's 67.7 ms AllGather p99
outlier. For four requests, the label-optimal DLO grouping is the wrong end-to-end
choice: it makes SP traffic cross the slower links and raises combined median by
9.67%. The measured planner correctly selects the natural mapping instead.

DP shapes represent different request concurrency and should not be ranked solely by
the sum in the table. A single MiniMax-H3 request must use DP1 x SP8; DP2/4/8 are
eligible only when at least 2/4/8 independent requests are available.

This result supports the planner's central policy: topology labels provide a safe
initial candidate, but measured DLO and SP collectives decide the physical mapping.
