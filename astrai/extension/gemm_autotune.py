"""Runtime plan-row autotuner: humming's prepare-time dispatch, AOT form.

humming (the JIT GEMM library this port follows) splits dispatch in two:
the heavy selection runs once at prepare time (per-layer configs ->
per-device heuristics -> one JIT cubin per shape-m interval), and the run
path is a single interval lookup in C++. The AOT port keeps AstrAI's
compiled-in kernels and moves the "prepare" half onto measured plan rows:

- coverage check: ``plan_probe`` (a host-only binding) says which row tier
  would serve a shape — ``builtin`` rows are the AOT baseline and
  ``override`` is the ``ASTR_GEMM_TABLE`` experimenter channel, both
  owners the tuner defers to. An ``injected`` answer is then resolved
  against the measured rows themselves (first-match, exactly as
  ``plan_row_for`` would): the heuristic floor installs as injected rows
  too, but it is a placeholder, not coverage — only a shape no measured
  row serves is the autotuner's to fill.
- top-up: a degraded shape seen enough times gets a one-time candidate
  sweep (``tile_vocabulary`` filtered to the staging pair, one injected row
  per candidate via ``set_plan_table_override``, interleaved CUDA-event
  medians over the caller's own tensors), and the winner is installed as
  an injected row — ranked below the env file, above the builtin table.
- persistence: winners live in ``~/.astrai/cache/gemm_plans/<device>.rows``
  keyed by the device geometry (SM count, smem, L2 — a band bound is
  device arithmetic, so a different part re-tunes). This is the layer
  humming itself lacks: its heuristics re-derive every process.

Deliberately NOT runtime-tuned: shapes the builtin table already serves.
Recalibrating those for a new part is the offline flow's job
(``csrc/bench/tune_baseline.py``); this module only fills holes the AOT
table structurally cannot.

Env knobs: ``ASTR_GEMM_AUTOTUNE=1`` enables the hook (or call
:func:`enable`); ``ASTR_GEMM_TUNE_DIR`` relocates the cache;
``ASTR_GEMM_TUNE_MAX_SHAPES`` caps tunes per process (default 32);
``ASTR_GEMM_TUNE_TRIGGER`` is the recurrence count before a degraded shape
tunes (default 3). While ``ASTR_GEMM_TABLE`` is set the tuner idles — the
env channel outranks injected rows, so candidate forcing would not take
and measurements would lie.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

from astrai.extension.loader import get_module

logger = logging.getLogger(__name__)

ENV_AUTOTUNE = "ASTR_GEMM_AUTOTUNE"
ENV_CACHE_DIR = "ASTR_GEMM_TUNE_DIR"
ENV_MAX_SHAPES = "ASTR_GEMM_TUNE_MAX_SHAPES"
ENV_TRIGGER = "ASTR_GEMM_TUNE_TRIGGER"

DEFAULT_MAX_SHAPES = 32
DEFAULT_TRIGGER = 3
_MEASURE_TRIALS = 5
_MEASURE_WARMUP = 3

# CTA geometry per TileClass ordinal — the row file's cta column contract
# (policy.cuh kTileClassCta, static_asserted against the instantiated tiles).
CTA_GEOMETRY = ((64, 64), (128, 64), (128, 128), (128, 256))

# GemmPerfClass -> operand widths (the width pair picks the manifest ladder,
# so the candidate space keys on it, not on the class id itself).
PERF_WIDTHS = {0: (2, 2), 1: (2, 1), 2: (1, 1), 3: (1, 1)}

# torch dtype -> the extension's operand vocabulary (find_gemm_dispatch).
_DTYPES = {
    "bfloat16": "bf16",
    "int8": "int8",
    "float8_e4m3fn": "fp8e4m3",
    "float8_e5m2": "fp8e5m2",
}


@dataclass(frozen=True)
class Problem:
    """The planner's dispatch key for one call (gemm.cuh PlanQuery)."""

    m: int
    n: int
    k: int
    batch: int
    perf_class: int
    crosswise: int


@dataclass(frozen=True)
class Row:
    """One injected plan row; the field order is the row-file order."""

    m_min: int
    m_max: int
    n_min: int
    n_max: int
    perf_class: int
    crosswise: int
    cta: int
    stages: int
    kk: int

    def text(self) -> str:
        # raster 0 = plan_raster at launch (the aspect heuristic owns it).
        return (
            f"{self.m_min} {self.m_max} {self.n_min} {self.n_max}"
            f" {self.perf_class} {self.crosswise} {self.cta}"
            f" {self.stages} 0 {self.kk}"
        )

    def recipe(self) -> Tuple[int, int, int]:
        return (self.cta, self.stages, self.kk)


def perf_class_of(dt_a: torch.dtype, dt_b: torch.dtype) -> int:
    """Mirror of find_gemm_dispatch's dtype-pair switch (gemm.cu)."""
    a = _DTYPES.get(str(dt_a).removeprefix("torch."))
    b = _DTYPES.get(str(dt_b).removeprefix("torch."))
    if (a, b) == ("bf16", "bf16"):
        return 0
    if (a, b) == ("bf16", "int8") or (a == "bf16" and b in ("fp8e4m3", "fp8e5m2")):
        return 1
    if (a, b) == ("int8", "int8"):
        return 2
    if a == b and a in ("fp8e4m3", "fp8e5m2"):
        return 3
    raise ValueError(f"unsupported operand dtype pair {dt_a} x {dt_b}")


def problem_of(
    a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool
) -> Problem:
    """The dispatch key the C++ side would derive for this call: the trans
    flags folded with any inner-transposed storage (resolve_operand's
    zero-copy fold — the storage flip swaps the layout tag the planner
    keys on), the shapes from the user flags, and the crosswise count from
    the folded tags (A ColMajor-tag or B RowMajor-tag each stage direct).
    """
    col_a = a.stride(-1) != 1 and a.stride(-2) == 1
    col_b = b.stride(-1) != 1 and b.stride(-2) == 1
    tag_a = trans_a ^ col_a
    tag_b = trans_b ^ col_b
    batch_a = a.size(0) if a.dim() == 3 else 1
    batch_b = b.size(0) if b.dim() == 3 else 1
    return Problem(
        m=a.size(-1) if trans_a else a.size(-2),
        n=b.size(-2) if trans_b else b.size(-1),
        k=a.size(-2) if trans_a else a.size(-1),
        batch=max(batch_a, batch_b),
        perf_class=perf_class_of(a.dtype, b.dtype),
        crosswise=(1 if tag_a else 0) + (0 if tag_b else 1),
    )


def ring_bytes(cta: int, stages: int, kk: int, ba: int, bb: int) -> int:
    """ring_smem_bytes (policy.cuh), for candidate feasibility."""
    bm, bn = CTA_GEOMETRY[cta]
    return (stages + 1) * kk * (bm * ba + bn * bb)


def device_signature(facts: dict) -> str:
    """Cache key from the device geometry a band bound prices against."""
    return f"cc{facts['cc']}-sms{facts['sms']}-smem{facts['smem_max']}-l2{facts['l2_bytes']}"


def heuristic_rows(facts: dict) -> List[Row]:
    """Crosswise first-fit for devices with no cached rows — humming's
    DeviceHeuristics shape, in AstrAI's recipe vocabulary: the degraded
    band ladder (small / narrow / big by M) with the ring depth raised
    where the device's smem leaves room. Crosswise layouts only: the
    builtin rows are all crosswise-0, so this covers exactly the shapes
    that would otherwise degrade, and shadows nothing measured.
    """
    rows: List[Row] = []
    for perf, (ba, bb) in PERF_WIDTHS.items():
        for crosswise in (1, 2):
            ladder = ((0, 512, 0), (512, 3072, 1), (3072, 0, 2))
            for m_min, m_max, cta in ladder:
                stages = 3 if cta == 0 else 2
                if ring_bytes(cta, stages, 64, ba, bb) > facts["smem_max"]:
                    stages = 2
                if ring_bytes(cta, stages, 64, ba, bb) > facts["smem_max"]:
                    cta, stages = 0, 2  # the small s2 floor every part fits
                rows.append(Row(m_min, m_max, 0, 0, perf, crosswise, cta, stages, 64))
    return rows


class GemmAutotuner:
    """The note() hook plus the one-time tune machinery behind it."""

    def __init__(self) -> None:
        self._mod: Optional[object] = None
        self._lock = threading.Lock()
        self._tuning = False
        self._active = False
        self._tiers: dict[Problem, str] = {}  # probe answers (diagnostics)
        self._covered: dict[Problem, bool] = {}  # the tune decision, cached
        self._misses: dict[Problem, int] = {}
        self._done: set = set()  # problems already tuned (or refused) here
        self._rows: List[Row] = []  # measured winners, newest first
        self._base_rows: List[Row] = []  # heuristic floor under them
        self._facts: Optional[dict] = None
        self._vocab: Optional[Sequence[Sequence[int]]] = None
        self._tuned = 0
        self._max_shapes = int(os.environ.get(ENV_MAX_SHAPES, DEFAULT_MAX_SHAPES))
        self._trigger = int(os.environ.get(ENV_TRIGGER, DEFAULT_TRIGGER))
        self._cache_path: Optional[Path] = None
        self._deadline = math.inf

    # -- module indirection so tests can substitute a fake ---------------
    def _gemm(self) -> object:
        if self._mod is None:
            self._mod = get_module("gemm")
        return self._mod

    def start(self, time_budget_s: float = 60.0) -> bool:
        """Load the persistent cache (or the heuristic floor) and install
        the rows. Returns False when the tuner cannot run (old extension
        build); with ASTR_GEMM_TABLE set the tuner idles by design.
        """
        mod = self._gemm()
        needed = (
            "plan_probe",
            "set_plan_table_override",
            "tile_vocabulary",
            "device_facts_info",
        )
        if not all(hasattr(mod, name) for name in needed):
            logger.warning(
                "gemm extension lacks the autotune bindings; "
                "rebuild with CSRC_KERNELS=true to enable ASTR_GEMM_AUTOTUNE"
            )
            return False
        if os.environ.get("ASTR_GEMM_TABLE"):
            logger.info("ASTR_GEMM_TABLE owns the plan table; the autotuner idles")
            self._active = True  # note() records, nothing tunes
            return True
        self._facts = mod.device_facts_info()
        self._vocab = mod.tile_vocabulary()
        cache_dir = Path(
            os.environ.get(ENV_CACHE_DIR, "~/.astrai/cache/gemm_plans")
        ).expanduser()
        self._cache_path = cache_dir / f"{device_signature(self._facts)}.rows"
        self._rows = self._load_rows(self._cache_path)
        self._base_rows = heuristic_rows(self._facts)
        self._install()
        if time_budget_s > 0:
            self._deadline = time.monotonic() + time_budget_s
        self._active = True
        logger.info(
            "gemm autotune on (%d cached rows, heuristic floor %d rows)",
            len(self._rows),
            len(self._base_rows),
        )
        return True

    # -- the hot-path hook ------------------------------------------------
    def note(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: Optional[torch.Tensor],
        b_scale: Optional[torch.Tensor],
        trans_a: bool,
        trans_b: bool,
        bias: Optional[torch.Tensor],
    ) -> None:
        """Called from the quant_gemm wrapper; never raises onto the call."""
        try:
            self._note(a, b, a_scale, b_scale, trans_a, trans_b, bias)
        except Exception:  # noqa: BLE001 — an optional tuner must not break a launch
            logger.exception("gemm autotune note() failed; disabling")
            self._active = False

    def _note(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: Optional[torch.Tensor],
        b_scale: Optional[torch.Tensor],
        trans_a: bool,
        trans_b: bool,
        bias: Optional[torch.Tensor],
    ) -> None:
        if self._tuning or not self._active:
            return
        prob = problem_of(a, b, trans_a, trans_b)
        covered = self._covered.get(prob)
        if covered is None:
            covered = self._covered[prob] = self._coverage(a, b, prob, trans_a, trans_b)
        if covered:
            return  # builtin / override / a measured row owns this shape
        if prob in self._done:
            return  # one tune attempt per problem per process
        if os.environ.get("ASTR_GEMM_TABLE"):
            return  # env owns the source even mid-process
        misses = self._misses.get(prob, 0) + 1
        self._misses[prob] = misses
        if misses >= self._trigger:
            self._tune(prob, a, b, a_scale, b_scale, trans_a, trans_b, bias)

    def _coverage(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        prob: Problem,
        trans_a: bool,
        trans_b: bool,
    ) -> bool:
        """Does anything OWN this shape? The builtin and env tables do; a
        measured row does. The heuristic floor does NOT — it is a
        placeholder until this shape tunes, and the probe cannot tell the
        two apart (both install as injected rows), so the injected tier is
        resolved by simulating the installed rows' first match: measured
        rows sit in front, so a first match among them is coverage."""
        tier = self._probe(a, b, prob, trans_a, trans_b)
        self._tiers[prob] = tier
        if tier in ("builtin", "override"):
            return True
        if tier != "injected":
            return False  # degraded: nothing installed matches either
        for row in self._rows:  # the measured rows, in installed order
            if self._row_matches(prob, row):
                return True
        return False

    @staticmethod
    def _row_matches(prob: Problem, row: Row) -> bool:
        """The row-file band predicate over the fields the tuner emits
        (open K, no gates): bands are (min, max], 0 = open."""
        if prob.m <= row.m_min or (row.m_max != 0 and prob.m > row.m_max):
            return False
        if prob.n <= row.n_min or (row.n_max != 0 and prob.n > row.n_max):
            return False
        return row.perf_class == prob.perf_class and row.crosswise == prob.crosswise

    def _probe(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        prob: Problem,
        trans_a: bool,
        trans_b: bool,
    ) -> str:
        """Which row tier would serve this shape. The probe receives the
        FOLDED trans tags (storage flips included), so the C++ side derives
        the same layout pair the real call instantiates with."""
        col_a = a.stride(-1) != 1 and a.stride(-2) == 1
        col_b = b.stride(-1) != 1 and b.stride(-2) == 1
        info = self._gemm().plan_probe(
            prob.m,
            prob.n,
            prob.k,
            a.dtype,
            b.dtype,
            trans_a ^ col_a,
            trans_b ^ col_b,
            prob.batch,
        )
        return str(info["source"])

    # -- tuning ------------------------------------------------------------
    def _candidates(self, prob: Problem) -> List[Row]:
        """The recipe vocabulary filtered to this staging pair and the
        device's smem ceiling — the compiled truth from tile_vocabulary,
        not a Python-side re-parse of policy.cuh."""
        ba, bb = PERF_WIDTHS[prob.perf_class]
        assert self._facts is not None
        out: List[Row] = []
        for entry in self._vocab or ():
            cw, vba, vbb, cta, stages, kk = entry
            if cw != (1 if prob.crosswise else 0):
                continue
            if (vba, vbb) != (ba, bb):
                continue
            if ring_bytes(cta, stages, kk, ba, bb) > self._facts["smem_max"]:
                continue
            out.append(
                Row(
                    prob.m - 1,
                    prob.m,
                    prob.n - 1,
                    prob.n,
                    prob.perf_class,
                    prob.crosswise,
                    cta,
                    stages,
                    kk,
                )
            )
        return out

    def _install(self) -> None:
        text = "\n".join(r.text() for r in self._rows + self._base_rows)
        self._gemm().set_plan_table_override(text)

    def _install_one(self, row: Row) -> None:
        """Force one candidate: it alone in front, nothing else to shadow it."""
        self._gemm().set_plan_table_override(row.text())

    def _tune(
        self,
        prob: Problem,
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: Optional[torch.Tensor],
        b_scale: Optional[torch.Tensor],
        trans_a: bool,
        trans_b: bool,
        bias: Optional[torch.Tensor],
    ) -> None:
        # One attempt per problem per process, whatever the outcome: a cap
        # hit, a budget expiry or an empty candidate list will not change on
        # retry, and a serving call must not re-enter a failed tune loop.
        self._done.add(prob)
        if self._tuned >= self._max_shapes or time.monotonic() > self._deadline:
            return
        candidates = self._candidates(prob)
        if not candidates:
            return
        with self._lock:
            if self._tuning:
                return
            self._tuning = True
        try:
            winner = self._measure(
                prob, candidates, a, b, a_scale, b_scale, trans_a, trans_b, bias
            )
            if winner is not None:
                self._merge(winner)
                self._install()
                self._persist()
                self._tuned += 1
                logger.info(
                    "gemm autotune %dx%dx%d b=%d class %d cw %d -> cta%d s%d k%d",
                    prob.m,
                    prob.n,
                    prob.k,
                    prob.batch,
                    prob.perf_class,
                    prob.crosswise,
                    *winner.recipe(),
                )
            # Refresh the coverage answer whatever happened: a winner's
            # grown band covers the shape, a failure leaves it uncovered
            # (and _done keeps the retry away either way).
            self._tiers[prob] = self._probe(a, b, prob, trans_a, trans_b)
            self._covered[prob] = any(self._row_matches(prob, r) for r in self._rows)
        finally:
            self._tuning = False

    def _measure(
        self,
        prob: Problem,
        candidates: List[Row],
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: Optional[torch.Tensor],
        b_scale: Optional[torch.Tensor],
        trans_a: bool,
        trans_b: bool,
        bias: Optional[torch.Tensor],
    ) -> Optional[Row]:
        """Interleaved candidate sweep, medians (workspace benchmark rules:
        one process, warmup, sync before/after each timed region)."""
        mod = self._gemm()

        def run() -> None:
            mod.quant_gemm(a, b, a_scale, b_scale, trans_a, trans_b, bias)

        times: List[List[float]] = [[] for _ in candidates]
        for row in candidates:
            self._install_one(row)
            for _ in range(_MEASURE_WARMUP):
                run()
        torch.cuda.synchronize()
        for _ in range(_MEASURE_TRIALS):
            for i, row in enumerate(candidates):
                self._install_one(row)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run()
                end.record()
                torch.cuda.synchronize()
                times[i].append(start.elapsed_time(end))
        medians = [sorted(t)[len(t) // 2] for t in times]
        best = medians.index(min(medians))
        return candidates[best]

    def _merge(self, winner: Row) -> None:
        """Interval growth, humming's get_configs compression: a winner
        generalizes one octave up in M (decode shapes arrive from below),
        and a later winner of the SAME recipe whose band touches an
        earlier one extends it instead of stacking a new row."""
        grown = Row(
            winner.m_min,
            winner.m_max + max(1, winner.m_max // 4),
            winner.n_min,
            winner.n_max + max(1, winner.n_max // 4),
            winner.perf_class,
            winner.crosswise,
            winner.cta,
            winner.stages,
            winner.kk,
        )
        for i, row in enumerate(self._rows):
            same_key = (
                row.recipe() == grown.recipe()
                and row.perf_class == grown.perf_class
                and row.crosswise == grown.crosswise
                and row.n_min <= grown.n_max
                and grown.n_min <= row.n_max
            )
            near = row.m_min <= grown.m_max + max(1, grown.m_max // 2)
            if same_key and near:
                self._rows[i] = Row(
                    min(row.m_min, grown.m_min),
                    max(row.m_max, grown.m_max),
                    min(row.n_min, grown.n_min),
                    max(row.n_max, grown.n_max),
                    row.perf_class,
                    row.crosswise,
                    row.cta,
                    row.stages,
                    row.kk,
                )
                return
        self._rows.insert(0, grown)  # newest measurement first (first match)

    # -- persistence -------------------------------------------------------
    def _load_rows(self, path: Path) -> List[Row]:
        if not path.is_file():
            return []
        rows: List[Row] = []
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = [int(x) for x in line.split()]
            if len(fields) != 10:
                logger.warning("%s: skipping row with %d fields", path, len(fields))
                continue
            # File field order: ... cta stages RASTER kk. The tuner always
            # emits auto-raster (0); anything else is a hand edit it does
            # not round-trip, so it is dropped with a note rather than
            # silently reinterpreted.
            m_min, m_max, n_min, n_max, perf, cw, cta, stages, raster, kk = fields
            if raster != 0:
                logger.warning("%s: dropping hand-set raster %d", path, raster)
            rows.append(Row(m_min, m_max, n_min, n_max, perf, cw, cta, stages, kk))
        return rows

    def _persist(self) -> None:
        assert self._cache_path is not None and self._facts is not None
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# gemm plan rows for {device_signature(self._facts)}\n"
            f"# measured winners only (runtime top-up); field order is the\n"
            f"# row-file order: m_min m_max n_min n_max perf crosswise cta\n"
            f"# stages raster kk — raster column omitted below (always auto)\n"
        )
        self._cache_path.write_text(
            header + "\n".join(r.text() for r in self._rows) + "\n"
        )


def enable(time_budget_s: float = 60.0) -> bool:
    """Install the autotune hook into the quant_gemm wrapper (the program
    ASTR_GEMM_AUTOTUNE=1 would install on first call)."""
    tuner = GemmAutotuner()
    if not tuner.start(time_budget_s):
        return False

    from astrai.extension.ops import gemm as ops_gemm

    ops_gemm.set_autotuner(tuner)
    return True
