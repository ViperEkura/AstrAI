"""The GEMM plan: one value, one writer, one scope — plus the runtime autotuner.

This is the gemm family's policy layer (the C++ config state calls itself
"the backing of astrai.extension.plan"). ``plan.config`` reads the whole
configuration as a value, ``plan.configure`` writes a patch and returns the
new value, ``plan.override`` scopes a patch for an A/B (restoring knobs *and*
rows, exception or not), and ``plan.probe`` / ``plan.facts`` / ``plan.tiles``
read the planner's own answers. Records answer to both idioms — attribute
access and the old ``d["key"]`` / ``row[3]`` / tuple unpacking.

The raw-dict views of the same bindings stay in
``astrai.extension.ops.gemm`` (``set_table`` / ``set_planner`` / ``set_log`` /
``set_staging`` / ``state`` / ``probe`` / ``facts`` / ``tile_vocabulary``):
the four ``csrc/bench`` tools parse those shapes, so those spellings are
contract. The autotuner at the bottom — and the per-launch hook the adapter
calls — live here too, because they are policy, not argument marshalling.
"""

import logging
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import torch

from astrai.extension.loader import get_module

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


# The autotune hook: the adapter calls note_launch on every launch, so the
# off path costs one flag check. Nothing is installed unless the deprecated
# ASTR_GEMM_AUTOTUNE=1 seed or enable() put a tuner there.
_autotuner: Optional[object] = None
_autotuner_resolved = False


def set_autotuner(tuner: object) -> None:
    """Install (or, with None, remove) the runtime autotune hook."""
    global _autotuner
    _autotuner = tuner


def note_launch(a, b, a_scale, b_scale, trans_a, trans_b, bias) -> None:
    """Record one quant_gemm launch for the installed tuner (the adapter's
    per-launch hook)."""
    global _autotuner_resolved
    if not _autotuner_resolved:
        _autotuner_resolved = True
        if os.environ.get("ASTR_GEMM_AUTOTUNE", "") == "1":  # deprecated seed
            enable()
    if _autotuner is not None:
        _autotuner.note(a, b, a_scale, b_scale, trans_a, trans_b, bias)


# Planner configuration — the runtime owner of every launch-time knob that
# used to live in environment variables (and, for the plan rows, in a
# hand-pasted plan_table.h block that needed a rebuild). Values set here
# win; anything unset falls to a one-time seed from the deprecated
# ASTR_GEMM_* variables, never to a per-call env read. Sweeps toggle these
# knobs between launches instead of rewriting the environment.
# ---------------------------------------------------------------------------

PLANNER_MODES = ("table", "hybrid", "model")
ROW_TIERS = ("override", "injected")

Rows = Union[str, Path]


# ---------------------------------------------------------------------------
# The plan surface: one value (``plan.config``), one writer
# (``plan.configure``), one scope (``plan.override``) — over the same bindings
# the legacy ``set_*`` / ``state`` / ``probe`` / ``facts`` functions call.
# Those keep returning the raw wire dict/list (the four csrc/bench tools parse
# those keys); the records here are for new code and stay answerable to the
# old idioms — attribute access, ``d["key"]``, ``row[3]`` and 12-tuple
# unpacking all work — so a call site migrates one line at a time.
# ---------------------------------------------------------------------------


class _Record:
    """A frozen dataclass that still reads like the dict/list it replaced."""

    __slots__ = ()

    def _values(self) -> tuple:
        return tuple(getattr(self, f.name) for f in fields(self))

    def __getitem__(self, key):
        return getattr(self, key) if isinstance(key, str) else self._values()[key]

    def __iter__(self):
        return iter(self._values())

    def __len__(self) -> int:
        return len(fields(self))

    def keys(self) -> tuple:
        return tuple(f.name for f in fields(self))

    def items(self) -> tuple:
        return tuple(zip(self.keys(), self._values()))

    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass(frozen=True)
class Staging(_Record):
    """The staging A/B switches; both default to enabled."""

    tma: bool
    mx: bool


@dataclass(frozen=True)
class PlanConfig(_Record):
    """The whole planner configuration as one value.

    ``planner_mode`` is the raw knob (-1 = unset: the one-time env seed
    decides) while ``planner`` is the resolved name. Each row tier carries the
    source spec it was installed from, which is what makes the value
    re-installable verbatim.
    """

    planner: str
    planner_mode: int
    log: bool
    table_off: bool
    override_rows: int
    override_source: str
    injected_rows: int
    injected_source: str
    staging: Staging

    @classmethod
    def _of(cls, wire: dict) -> "PlanConfig":
        table = wire["table"]
        return cls(
            planner=wire["planner"],
            planner_mode=wire["planner_mode"],
            log=wire["log"],
            table_off=table["off"],
            override_rows=table["override_rows"],
            override_source=table["override_source"],
            injected_rows=table["injected_rows"],
            injected_source=table["injected_source"],
            staging=Staging(**wire["staging"]),
        )


@dataclass(frozen=True)
class Decision(_Record):
    """The dispatch decision for one problem: ``source`` names the row tier
    (or the model / degraded end) that answered."""

    source: str
    cta: int
    stages: int
    raster: int
    kk: int
    perf_class: int
    crosswise: int


@dataclass(frozen=True)
class DeviceFacts(_Record):
    """The device geometry the planner prices against."""

    sms: int
    smem_max: int
    smem_per_sm: int
    regs_per_sm: int
    l2_bytes: int
    cc: int


@dataclass(frozen=True)
class Tile(_Record):
    """One ladder recipe. The field order is the vocabulary row's, so
    ``row[3]`` is still the CTA class and a 12-tuple still unpacks."""

    crosswise: int
    ba: int
    bb: int
    cta: int
    stages: int
    kk: int
    bm: int
    bn: int
    wm: int
    wn: int
    threads: int
    smem: int

    @property
    def cta_name(self) -> str:
        return _tile_class_names()[self.cta]

    @property
    def name(self) -> str:
        return f"Tile_{self.bm}x{self.bn}x{self.kk}_W{self.wm}x{self.wn}_S{self.stages}"


_TILE_NAMES: Optional[Tuple[str, ...]] = None


def _tile_class_names() -> Tuple[str, ...]:
    """The CTA class names in ordinal order (static for the process)."""
    global _TILE_NAMES
    if _TILE_NAMES is None:
        _TILE_NAMES = tuple(get_module("gemm").tile_class_names())
    return _TILE_NAMES


# This module IS the plan surface — read it as one value, write it with one
# call, scope it for an A/B:
#
#     cfg = plan.config                      # the whole state
#     plan.configure(planner="model", tma=False)
#     with plan.override(rows=rows, tier="override"):
#         ...                                # restored on exit, always
#
# Every configure() argument defaults to "leave unchanged", and the value it
# returns is fully re-installable — feeding a saved config's fields back
# restores that exact state (both row tiers ride their source spec).


def configure(
    planner: Union[str, int, None] = None,
    log: Optional[bool] = None,
    tma: Optional[bool] = None,
    mx: Optional[bool] = None,
    table_off: Optional[bool] = None,
    rows: Optional[Rows] = None,
    tier: Optional[str] = None,
) -> PlanConfig:
    """Apply a patch; returns the resulting configuration.

    Every argument defaults to "leave it alone", and only the ones given ride
    the patch: the binding takes a dict keyed by these names, so an unset
    argument is an absent key rather than a second spelling of "no change".

    ``planner`` is a mode name, an int, or ``""`` to restore the unset state
    (the env seed decides). ``rows`` is a row-file path or inline row text;
    the empty string clears the tier named by ``tier`` (``"override"``, the
    default, or ``"injected"``).
    """
    if tier is not None and tier not in ROW_TIERS:
        raise ValueError(f"tier must be one of {ROW_TIERS}, got {tier!r}")
    if isinstance(planner, str) and planner and planner not in PLANNER_MODES:
        raise ValueError(f"planner must be one of {PLANNER_MODES}, got {planner!r}")
    patch = {
        "planner": planner,
        "log": log,
        "tma": tma,
        "mx": mx,
        "table_off": table_off,
        "rows": str(rows) if isinstance(rows, Path) else rows,
        "tier": tier,
    }
    wire = get_module("gemm").configure(
        {key: value for key, value in patch.items() if value is not None}
    )
    return PlanConfig._of(wire)


def _config() -> PlanConfig:
    return PlanConfig._of(get_module("gemm").config_state())


def _facts() -> DeviceFacts:
    return DeviceFacts(**get_module("gemm").device_facts_info())


@contextmanager
def override(**patch) -> Iterator[PlanConfig]:
    """Apply a patch for the duration of the block, then restore.

    The restore re-installs the saved value wholesale, so the knobs and both
    row tiers come back — whether the block returns or raises.
    """
    saved = _config()
    configure(**patch)
    try:
        yield _config()
    finally:
        configure(
            planner=saved.planner_mode,
            log=saved.log,
            tma=saved.staging.tma,
            mx=saved.staging.mx,
            table_off=saved.table_off,
            rows=saved.override_source,
            tier="override",
        )
        configure(rows=saved.injected_source, tier="injected")


def probe(
    m: int,
    n: int,
    k: int,
    dt_a: torch.dtype = torch.bfloat16,
    dt_b: torch.dtype = torch.bfloat16,
    trans_a: bool = False,
    trans_b: bool = True,
    batch: int = 1,
) -> Decision:
    """The dispatch decision for one problem — GPU-free (no launch)."""
    return Decision(
        **get_module("gemm").plan_probe(
            m, n, k, dt_a, dt_b, trans_a=trans_a, trans_b=trans_b, batch=batch
        )
    )


def tiles() -> List[Tile]:
    """Every recipe the launch ladders instantiate, each carrying its CTA
    class name and canonical ``Tile_...`` spelling."""
    return [Tile(*row) for row in get_module("gemm").tile_vocabulary()]


def __getattr__(name: str):  # PEP 562: the two read-only values of the surface
    if name == "config":
        return _config()
    if name == "facts":
        return _facts()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    """The planner's dispatch key for one call (gemm.cuh PlanQuery).

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


def device_signature(facts: dict) -> str:
    """Cache key from the device geometry a band bound prices against."""
    return (
        f"cc{facts['cc']}-sms{facts['sms']}"
        f"-smem{facts['smem_max']}-l2{facts['l2_bytes']}"
    )


def heuristic_rows(facts: dict, vocab, width_perf: dict) -> List[Row]:
    """Crosswise first-fit for devices with no cached rows — humming's
    DeviceHeuristics shape: the degraded band ladder (small / narrow / big
    by M) with the ring depth raised where the device's smem leaves room.
    Crosswise layouts only (the builtin rows are all crosswise-0), so this
    covers exactly the shapes that would otherwise degrade and shadows
    nothing measured. smem per recipe comes from the vocabulary and the
    perf column from the probe-derived ``width_perf`` mapping (the perf
    class doubles as the width constraint: the two are 1:1), so nothing
    here mirrors policy.cuh or gemm.cu by hand.
    """
    ring = {
        (entry[3], entry[4], entry[5], entry[1], entry[2]): entry[9] for entry in vocab
    }  # (cta, stages, kk, ba, bb) -> ring smem, from tile_vocabulary
    rows: List[Row] = []
    for (ba, bb), perf in sorted(width_perf.items()):
        for crosswise in (1, 2):
            ladder = ((0, 512, 0), (512, 3072, 1), (3072, 0, 2))
            for m_min, m_max, cta in ladder:
                stages = 3 if cta == 0 else 2
                if ring.get((cta, stages, 64, ba, bb), 1 << 30) > facts["smem_max"]:
                    stages = 2
                if ring.get((cta, stages, 64, ba, bb), 1 << 30) > facts["smem_max"]:
                    cta, stages = 0, 2  # the small s2 floor every part fits
                rows.append(Row(m_min, m_max, 0, 0, perf, crosswise, cta, stages, 64))
    return rows


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
        self._tiers: dict = {}  # probe answers (diagnostics)
        self._perf_of: dict = {}  # dtype pair -> perf class, from the probe
        self._width_perf: dict = {}  # widths -> perf class, joined
        self._covered: dict = {}  # the tune decision, cached on the cheap key
        self._misses: dict = {}
        self._done: set = set()  # problems already tuned (or refused) here
        self._rows: List[Row] = []  # measured winners, newest first
        self._base_rows: List[Row] = []  # heuristic floor under them
        self._facts: Optional[dict] = None
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
        """Load the persistent cache (or the heuristic floor) and install
        the rows. Returns False when the tuner cannot run (old extension
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
            except Exception:  # noqa: BLE001 — an unsupported pair just has no floor
                continue
            self._perf_of[(dt_a, dt_b)] = int(info["perf_class"])
            self._width_perf[_WIDTHS_OF[(dt_a, dt_b)]] = int(info["perf_class"])
        cache_dir = Path(
            self._cache_dir
            or os.environ.get(ENV_CACHE_DIR, "~/.astrai/cache/gemm_plans")
        ).expanduser()
        self._cache_path = cache_dir / f"{device_signature(self._facts)}.rows"
        self._rows = self._load_rows(self._cache_path)
        self._base_rows = heuristic_rows(self._facts, self._vocab, self._width_perf)
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
        key: tuple,
        trans_a: bool,
        trans_b: bool,
    ) -> Optional[bool]:
        """Does anything OWN this shape? The builtin and override tables
        do; a measured row does. The heuristic floor does NOT — it is a
        placeholder until this shape tunes, and the probe cannot tell the
        two apart (both install as injected rows), so the injected tier is
        resolved by simulating the installed rows' first match: measured
        rows sit in front, so a first match among them is coverage. None
        means "not even a recipe exists for this pair" (treat as covered:
        nothing to tune).
        """
        info = self._probe(a, b, key, trans_a, trans_b)
        tier = str(info["source"])
        self._tiers[key] = tier
        self._perf_of[(key[5], key[6])] = int(info["perf_class"])
        prob = Problem(*key[:3], key[3], int(info["perf_class"]), key[4])
        if tier in ("builtin", "override", "model"):
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
        key: tuple,
        trans_a: bool,
        trans_b: bool,
    ) -> dict:
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
    def _candidates(self, key: tuple) -> List[Row]:
        """The recipe vocabulary filtered to this staging pair and the
        device's smem ceiling — the compiled truth from tile_vocabulary,
        not a Python-side re-parse of policy.cuh."""
        m, n, _, _, crosswise, dt_a, dt_b = key
        widths = _WIDTHS_OF.get((str(dt_a), str(dt_b)))
        assert self._facts is not None
        out: List[Row] = []
        if widths is None:
            return out
        perf = self._perf_of[(str(dt_a), str(dt_b))]  # cached by _coverage
        for entry in self._vocab or ():
            cw, vba, vbb, cta, stages, kk, _, _, _, smem = entry
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
                    stages,
                    kk,
                )
            )
        return out

    def _install(self) -> None:
        self._install_rows("\n".join(r.text() for r in self._rows + self._base_rows))

    def _install_one(self, row: Row) -> None:
        """Force one candidate: it alone in front, nothing else to shadow it."""
        self._install_rows(row.text())

    def _install_rows(self, text: str) -> None:
        """Install rows at the injected tier — below any user override, above
        the compiled-in tables."""
        self._gemm().configure({"rows": text, "tier": "injected"})

    def _tune(
        self,
        key: tuple,
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
        key: tuple,
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


def _problem_key(
    a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool
) -> tuple:
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
    set_autotuner(tuner)
    return True


__all__ = [
    "PlanConfig",
    "Staging",
    "Decision",
    "DeviceFacts",
    "Tile",
    "PLANNER_MODES",
    "ROW_TIERS",
    "Rows",
    "GemmAutotuner",
    "Problem",
    "Row",
    "device_signature",
    "heuristic_rows",
    "set_autotuner",
    "note_launch",
    "enable",
]
