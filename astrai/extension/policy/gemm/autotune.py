"""Runtime GEMM autotuner: candidate measurement, row cache, and launch hook setup."""

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from astrai.extension.runtime.loader import get_module

logger = logging.getLogger(__name__)


# The staging width pair per dtype pair (the vocabulary keys on it); the
# perf class itself always comes back from the probe. Keys are the torch
# dtype pair spellings the wrapper sees; the symmetric fp8 pairs ride the
# 1-byte ladders.
_WIDTHS_OF = {
    ("torch.bfloat16", "torch.bfloat16"): (2, 2),
    ("torch.bfloat16", "torch.int8"): (2, 1),
    ("torch.int8", "torch.int8"): (1, 1),
    ("torch.float8_e4m3fn", "torch.float8_e4m3fn"): (1, 1),
    ("torch.float8_e5m2", "torch.float8_e5m2"): (1, 1),
}


def _dtype_of(name: str):
    """The torch dtype for a _WIDTHS_OF key spelling."""
    return getattr(torch, name.removeprefix("torch."))


# ---------------------------------------------------------------------------
# Runtime autotuner: humming's prepare-time dispatch, AOT form. The run
# path stays a row lookup in C++; this half only fills holes the AOT table
# structurally cannot — shapes nothing else owns get a one-time candidate
# sweep over the recipe vocabulary (one injected row per candidate,
# interleaved CUDA-event medians over the caller's own tensors), and the
# winner installs as an injected row (below set_table overrides, above the
# builtin table) and persists under
# ``~/.astrai/cache/gemm_plans/<device>.rows``.
#
# Deliberately NOT runtime-tuned: shapes the builtin table or a set_table
# override already serves, and any shape while an override table owns the
# source (candidate forcing would not take and measurements would lie).
# ---------------------------------------------------------------------------

ENV_CACHE_DIR = "ASTR_GEMM_TUNE_DIR"
ENV_MAX_SHAPES = "ASTR_GEMM_TUNE_MAX_SHAPES"
ENV_TRIGGER = "ASTR_GEMM_TUNE_TRIGGER"

DEFAULT_MAX_SHAPES = 32
DEFAULT_TRIGGER = 3
_MEASURE_TRIALS = 5
_MEASURE_WARMUP = 3

# The config-patch schema the binding must speak (its own CONFIG_API attr).
# A stale .so is otherwise indistinguishable from a current one: its
# ``configure`` exists, it just takes keywords instead of a patch dict.
_CONFIG_API = 2


@dataclass(frozen=True)
class Problem:
    """The planner's dispatch key for one call (api/native/gemm_plan.h PlanQuery).

    perf_class comes from the probe (the C++ derivation), so this module
    keeps no dtype-pair mirror; the coverage cache keys on the cheap part
    (shapes + crosswise + dtypes) before any probe runs.
    """

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
    k_stages: int
    k_tile: int

    def text(self) -> str:
        # raster 0 = plan_raster at launch (the aspect heuristic owns it).
        return (
            f"{self.m_min} {self.m_max} {self.n_min} {self.n_max}"
            f" {self.perf_class} {self.crosswise} {self.cta}"
            f" {self.k_stages} 0 {self.k_tile}"
        )

    def recipe(self) -> Tuple[int, int, int]:
        return (self.cta, self.k_stages, self.k_tile)


def device_signature(facts: Dict) -> str:
    """Cache key from the device geometry a band bound prices against."""
    return (
        f"cc{facts['cc']}-sms{facts['sms']}"
        f"-smem{facts['smem_max']}-l2{facts['l2_bytes']}"
    )


class GemmAutotuner:
    """The note() hook plus the one-time tune machinery behind it."""

    def __init__(
        self,
        max_shapes: Optional[int] = None,
        trigger: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self._mod: Optional[object] = None
        self._lock = threading.Lock()
        self._tuning = False
        self._active = False
        self._tiers: Dict = {}  # probe answers (diagnostics)
        self._perf_of: Dict = {}  # dtype pair -> perf class, from the probe
        self._covered: Dict = {}  # the tune decision, cached on the cheap key
        self._misses: Dict = {}
        self._done: Set = set()  # problems already tuned (or refused) here
        self._rows: List[Row] = []  # measured winners, newest first
        self._facts: Optional[Dict] = None
        self._vocab: Optional[Sequence[Sequence[int]]] = None
        self._tuned = 0
        self._max_shapes = int(
            os.environ.get(ENV_MAX_SHAPES, DEFAULT_MAX_SHAPES)
            if max_shapes is None
            else max_shapes
        )
        self._trigger = int(
            os.environ.get(ENV_TRIGGER, DEFAULT_TRIGGER) if trigger is None else trigger
        )
        self._cache_dir = cache_dir
        self._cache_path: Optional[Path] = None
        self._deadline = math.inf

    # -- module indirection so tests can substitute a fake ---------------
    def _gemm(self) -> object:
        if self._mod is None:
            self._mod = get_module("gemm")
        return self._mod

    def start(self, time_budget_s: float = 60.0) -> bool:
        """Load and install measured rows from the persistent cache. Returns False when the tuner cannot run (old extension
        build); while an override table owns the source the tuner idles
        by design.
        """
        mod = self._gemm()
        needed = (
            "plan_probe",
            "configure",
            "config_state",
            "tile_vocabulary",
            "device_facts_info",
        )
        if not all(hasattr(mod, name) for name in needed) or (
            getattr(mod, "CONFIG_API", 1) < _CONFIG_API
        ):
            logger.warning(
                "gemm extension lacks the autotune bindings (or predates the "
                "plan's config patch schema); rebuild with CSRC_KERNELS=true "
                "to enable the autotuner"
            )
            return False
        table = mod.config_state()["table"]
        if table["off"] or table["override_rows"] > 0:
            logger.info("an override table owns the plan source; the autotuner idles")
            self._active = True  # note() records, nothing tunes
            return True
        self._facts = mod.device_facts_info()
        self._vocab = mod.tile_vocabulary()
        # width -> perf class: one host-only probe per supported dtype pair
        # (the probe IS the C++ derivation; nothing here mirrors it).
        for dt_a, dt_b in _WIDTHS_OF:
            try:
                info = mod.plan_probe(
                    1, 1, 1, _dtype_of(dt_a), _dtype_of(dt_b), False, True, 1
                )
            except Exception:  # noqa: BLE001 — skip unsupported pairs
                continue
            self._perf_of[(dt_a, dt_b)] = int(info["perf_class"])
        cache_dir = Path(
            self._cache_dir
            or os.environ.get(ENV_CACHE_DIR, "~/.astrai/cache/gemm_plans")
        ).expanduser()
        self._cache_path = cache_dir / f"{device_signature(self._facts)}.rows"
        self._rows = self._load_rows(self._cache_path)
        self._install()
        if time_budget_s > 0:
            self._deadline = time.monotonic() + time_budget_s
        self._active = True
        logger.info("gemm autotune on (%d cached rows)", len(self._rows))
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
        key = _problem_key(a, b, trans_a, trans_b)
        covered = self._covered.get(key)
        if covered is None:
            covered = self._covered[key] = self._coverage(a, b, key, trans_a, trans_b)
        if covered is not False:
            return  # builtin / override / a measured row owns this shape
        if key in self._done:
            return  # one tune attempt per problem per process
        table = self._gemm().config_state()["table"]
        if table["off"] or table["override_rows"] > 0:
            return  # an override table owns the source even mid-process
        misses = self._misses.get(key, 0) + 1
        self._misses[key] = misses
        if misses >= self._trigger:
            self._tune(key, a, b, a_scale, b_scale, trans_a, trans_b, bias)

    def _coverage(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        key: Tuple,
        trans_a: bool,
        trans_b: bool,
    ) -> Optional[bool]:
        """Measured rows own their covered shapes. Geometry-heuristic
        decisions remain eligible for tuning when the tuner is enabled.
        """
        info = self._probe(a, b, key, trans_a, trans_b)
        tier = str(info["source"])
        self._tiers[key] = tier
        self._perf_of[(key[5], key[6])] = int(info["perf_class"])
        prob = Problem(*key[:3], key[3], int(info["perf_class"]), key[4])
        if tier in ("builtin", "override", "model"):
            return True
        if tier != "injected":
            return False  # no measured row covers this shape
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
        key: Tuple,
        trans_a: bool,
        trans_b: bool,
    ) -> Dict:
        """The dispatch decision for this shape. The probe receives the
        FOLDED trans tags (storage flips included), so the C++ side derives
        the same layout pair the real call instantiates with."""
        m, n, k, batch, crosswise, _, _ = key
        col_a = a.stride(-1) != 1 and a.stride(-2) == 1
        col_b = b.stride(-1) != 1 and b.stride(-2) == 1
        return self._gemm().plan_probe(
            m, n, k, a.dtype, b.dtype, trans_a ^ col_a, trans_b ^ col_b, batch
        )

    # -- tuning ------------------------------------------------------------
    def _candidates(self, key: Tuple) -> List[Row]:
        """The recipe vocabulary filtered to this staging pair and the
        device's smem ceiling — the compiled truth from tile_vocabulary,
        not a Python-side re-parse of policy/gemm.cuh."""
        m, n, _, _, crosswise, dt_a, dt_b = key
        widths = _WIDTHS_OF.get((str(dt_a), str(dt_b)))
        assert self._facts is not None
        out: List[Row] = []
        if widths is None:
            return out
        perf = self._perf_of[(str(dt_a), str(dt_b))]  # cached by _coverage
        for entry in self._vocab or ():
            cw, vba, vbb, cta, k_stages, k_tile, _, _, _, smem = entry
            if cw != (1 if crosswise else 0):
                continue
            if (vba, vbb) != widths:
                continue
            if smem > self._facts["smem_max"]:
                continue
            out.append(
                Row(
                    m - 1,
                    m,
                    n - 1,
                    n,
                    perf if perf is not None else -1,
                    crosswise,
                    cta,
                    k_stages,
                    k_tile,
                )
            )
        return out

    def _install(self) -> None:
        self._install_rows("\n".join(r.text() for r in self._rows))

    def _install_one(self, row: Row) -> None:
        """Force one candidate: it alone in front, nothing else to shadow it."""
        self._install_rows(row.text())

    def _install_rows(self, text: str) -> None:
        """Install rows at the injected tier — below any user override, above
        the compiled-in tables."""
        self._gemm().configure({"rows": text, "tier": "injected"})

    def _tune(
        self,
        key: Tuple,
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
        self._done.add(key)
        if self._tuned >= self._max_shapes or time.monotonic() > self._deadline:
            return
        candidates = self._candidates(key)
        if not candidates:
            return
        with self._lock:
            if self._tuning:
                return
            self._tuning = True
        try:
            winner = self._measure(
                key, candidates, a, b, a_scale, b_scale, trans_a, trans_b, bias
            )
            if winner is not None:
                self._merge(winner)
                self._install()
                self._persist()
                self._tuned += 1
                m, n, k, batch, crosswise, _, _ = key
                logger.info(
                    "gemm autotune %dx%dx%d b=%d cw %d -> cta%d s%d k%d",
                    m,
                    n,
                    k,
                    batch,
                    crosswise,
                    *winner.recipe(),
                )
            # Refresh the coverage answer whatever happened: a winner's
            # grown band covers the shape, a failure leaves it uncovered
            # (and _done keeps the retry away either way).
            info = self._probe(a, b, key, trans_a, trans_b)
            self._tiers[key] = str(info["source"])
            prob = Problem(*key[:3], key[3], int(info["perf_class"]), key[4])
            self._covered[key] = any(self._row_matches(prob, r) for r in self._rows)
        finally:
            self._tuning = False

    def _measure(
        self,
        key: Tuple,
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
            winner.k_stages,
            winner.k_tile,
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
                    row.k_stages,
                    row.k_tile,
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
            # File field order: ... cta k_stages RASTER k_tile. The tuner always
            # emits auto-raster (0); anything else is a hand edit it does
            # not round-trip, so it is dropped with a note rather than
            # silently reinterpreted.
            (
                m_min,
                m_max,
                n_min,
                n_max,
                perf,
                cw,
                cta,
                k_stages,
                raster,
                k_tile,
            ) = fields
            if raster != 0:
                logger.warning("%s: dropping hand-set raster %d", path, raster)
            rows.append(
                Row(m_min, m_max, n_min, n_max, perf, cw, cta, k_stages, k_tile)
            )
        return rows

    def _persist(self) -> None:
        assert self._cache_path is not None and self._facts is not None
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# gemm plan rows for {device_signature(self._facts)}\n"
            f"# measured winners only (runtime top-up); field order is the\n"
            f"# row-file order: m_min m_max n_min n_max perf crosswise cta\n"
            f"# k_stages raster k_tile — raster column omitted below (always auto)\n"
        )
        self._cache_path.write_text(
            header + "\n".join(r.text() for r in self._rows) + "\n"
        )


def _problem_key(
    a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool
) -> Tuple:
    """The cheap pre-probe key: shapes, the folded crosswise count and the
    dtypes (the perf class it implies comes from the probe, never a
    Python-side mirror of the C++ switch)."""
    col_a = a.stride(-1) != 1 and a.stride(-2) == 1
    col_b = b.stride(-1) != 1 and b.stride(-2) == 1
    tag_a = trans_a ^ col_a
    tag_b = trans_b ^ col_b
    batch_a = a.size(0) if a.dim() == 3 else 1
    batch_b = b.size(0) if b.dim() == 3 else 1
    return (
        a.size(-1) if trans_a else a.size(-2),
        b.size(-2) if trans_b else b.size(-1),
        a.size(-2) if trans_a else a.size(-1),
        max(batch_a, batch_b),
        (1 if tag_a else 0) + (0 if tag_b else 1),
        str(a.dtype),
        str(b.dtype),
    )


def enable(
    time_budget_s: float = 60.0,
    max_shapes: Optional[int] = None,
    trigger: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> bool:
    """Install the autotune hook into the quant_gemm wrapper (the program
    ASTR_GEMM_AUTOTUNE=1 would install on first call). The keyword
    arguments default to the ASTR_GEMM_TUNE_* env seeds."""
    tuner = GemmAutotuner(max_shapes, trigger, cache_dir)
    if not tuner.start(time_budget_s):
        return False
    # plan re-exports this module; resolve its hook after initialization.
    from astrai.extension.policy.gemm.plan import set_autotuner

    set_autotuner(tuner)
    return True
