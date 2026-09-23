"""Interleaved A/B benchmark: full-tensor vs chunked no-grad logprobs.

Real AstrAI-1B shapes (vocab 100000, hidden 1536, 24 layers, 4 KV heads).
Full path is forced by hiding lm_head behind a wrapper (get_logprobs then
falls back); chunked path engages automatically under no_grad. Interleaved
rounds with alternating start side; medians reported. Peak transient memory
measured as max_memory_allocated delta around the call.
"""

import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from astrai.model.transformer import AutoRegressiveLM
from astrai.trainer.strategy import get_logprobs
from tests.helpers import make_tiny_config

DEV = "cuda:0"
VOCAB = 100000
torch.manual_seed(0)

cfg = make_tiny_config(
    vocab_size=VOCAB,
    hidden_size=1536,
    intermediate_size=6912,
    num_hidden_layers=24,
    num_attention_heads=24,
    num_key_value_heads=4,
    max_position_embeddings=8192,
)
model = AutoRegressiveLM(cfg).to(device=DEV, dtype=torch.bfloat16)
model.eval()
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B bf16")


class _NoLmHead(torch.nn.Module):
    """Hides lm_head so get_logprobs takes the full-tensor path."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids, attn_mask=None):
        return self.inner(input_ids, attn_mask)


full_model = _NoLmHead(model)


def make_batch(n, s):
    input_ids = torch.randint(3, VOCAB, (n, s), device=DEV)
    mask = torch.ones(n, s, dtype=torch.bool, device=DEV)
    return input_ids, mask, mask.clone()


def run_once(m, batch):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        get_logprobs(m, *batch, "none")
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def peak_mem(m, batch):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        get_logprobs(m, *batch, "none")
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


# numerical gate at real scale, once
b = make_batch(4, 2048)
with torch.no_grad():
    chunked = get_logprobs(model, *b, "none")["logprobs"]
    full = get_logprobs(full_model, *b, "none")["logprobs"]
diff = (chunked - full).abs().max().item()
print(f"max |chunked - full| at real scale: {diff:.3e}")

ROUNDS = 7
for n, s in [(4, 2048), (16, 2048), (32, 2048)]:
    batch = make_batch(n, s)
    try:
        for _ in range(3):
            run_once(model, batch)
            run_once(full_model, batch)
        ta, tb = [], []
        for r in range(ROUNDS):
            if r % 2 == 0:
                ta.append(run_once(full_model, batch))
                tb.append(run_once(model, batch))
            else:
                tb.append(run_once(model, batch))
                ta.append(run_once(full_model, batch))
        ma, mb = statistics.median(ta), statistics.median(tb)
        time_note = f"full {ma * 1e3:7.1f}ms chunked {mb * 1e3:7.1f}ms | time full/chunked {ma / mb:.2f}x"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        for _ in range(3):
            run_once(model, batch)
        tb = [run_once(model, batch) for _ in range(ROUNDS)]
        mb = statistics.median(tb)
        time_note = f"full OOM | chunked {mb * 1e3:7.1f}ms"
    try:
        pa = peak_mem(full_model, batch)
        full_note = f"{pa / 2**30:.2f}GB"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        full_note = "OOM"
        pa = None
    pb = peak_mem(model, batch)
    ratio = f"{pa / pb:.1f}x" if pa is not None else "inf (full OOM)"
    print(
        f"[{n:>2},{s}] {time_note}"
        f" | peak-transient full {full_note:>9} chunked {pb / 2**30:.2f}GB ({ratio})"
    )
