"""Runtime plan-row autotuner unit tests.

The tune machinery is tested against a fake ``gemm`` module (the planner
bindings as call records), so these run without a GPU or a built
extension: what is under test is the hook's coverage logic (which shapes
tune, which never do), the candidate filter, the winner's interval
growth, and the per-device persistence roundtrip.
"""

import pytest
import torch

from astrai.extension.gemm_autotune import (
    CTA_GEOMETRY,
    GemmAutotuner,
    Row,
    device_signature,
    heuristic_rows,
    perf_class_of,
    problem_of,
    ring_bytes,
)

FACTS = {
    "sms": 128,
    "smem_max": 101376,
    "smem_per_sm": 102400,
    "regs_per_sm": 65536,
    "l2_bytes": 75497472,
    "cc": 89,
}

# The cross six as tile_vocabulary rows (cw, ba, bb, cta, stages, kk).
VOCAB = [
    [0, 2, 2, 0, 2, 64],
    [0, 2, 2, 0, 3, 64],
    [0, 2, 2, 1, 2, 64],
    [0, 2, 2, 1, 3, 64],
    [0, 2, 2, 2, 2, 64],
    [0, 2, 2, 2, 3, 64],
    [0, 2, 2, 0, 2, 32],
    [0, 2, 2, 2, 2, 32],
    [1, 2, 2, 0, 2, 64],
    [1, 2, 2, 0, 3, 64],
    [1, 2, 2, 1, 2, 64],
    [1, 2, 2, 1, 3, 64],
    [1, 2, 2, 2, 2, 64],
    [1, 2, 2, 2, 3, 64],
    [0, 1, 1, 3, 2, 64],  # wide, 1-byte only
]


class FakeGemm:
    """The planner bindings as call records."""

    def __init__(self, source: str = "degraded"):
        self.source = source
        self.installs: list[str] = []
        self.probes = 0

    def plan_probe(self, m, n, k, dt_a, dt_b, trans_a, trans_b, batch=1):
        self.probes += 1
        return {
            "source": self.source,
            "cta": 0,
            "stages": 2,
            "raster": 0,
            "kk": 64,
            "perf_class": 0,
            "crosswise": 0,
        }

    def set_plan_table_override(self, source):
        self.installs.append(source)
        return source.count("\n") + 1

    def tile_vocabulary(self):
        return VOCAB

    def device_facts_info(self):
        return dict(FACTS)

    def quant_gemm(self, *args, **kwargs):
        return None


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("ASTR_GEMM_TABLE", raising=False)
    monkeypatch.delenv("ASTR_GEMM_AUTOTUNE", raising=False)
    monkeypatch.setenv("ASTR_GEMM_TUNE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ASTR_GEMM_TUNE_TRIGGER", "1")
    monkeypatch.setenv("ASTR_GEMM_TUNE_MAX_SHAPES", "8")


def nt_operands(m=64, n=128, k=256):
    # trans_b=True takes B stored [N][K] (the weight / fused-linear form).
    a = torch.zeros(m, k, dtype=torch.bfloat16)
    b = torch.zeros(n, k, dtype=torch.bfloat16)
    return a, b


class TestProblemDerivation:
    def test_nt_production_route(self):
        a, b = nt_operands()
        prob = problem_of(a, b, trans_a=False, trans_b=True)
        assert (prob.m, prob.n, prob.k) == (64, 128, 256)
        assert prob.perf_class == 0
        assert prob.crosswise == 0  # dual-congruous, the fused-linear shape

    def test_tt_is_crosswise(self):
        a = torch.zeros(256, 64, dtype=torch.bfloat16)  # [K][M]
        b = torch.zeros(128, 256, dtype=torch.bfloat16)  # [N][K]
        prob = problem_of(a, b, trans_a=True, trans_b=True)
        assert prob.crosswise == 1
        assert (prob.m, prob.n, prob.k) == (64, 128, 256)

    def test_tn_is_dual_crosswise(self):
        a = torch.zeros(256, 64, dtype=torch.bfloat16)  # [K][M]
        b = torch.zeros(256, 128, dtype=torch.bfloat16)  # [K][N]
        prob = problem_of(a, b, trans_a=True, trans_b=False)
        assert prob.crosswise == 2
        assert (prob.m, prob.n, prob.k) == (64, 128, 256)

    def test_col_major_view_folds_the_tag(self):
        base = torch.zeros(64, 256, dtype=torch.bfloat16)
        a = base.t()  # [256][64] storage, col-major view
        assert a.stride(-1) != 1 and a.stride(-2) == 1
        b = torch.zeros(128, 256, dtype=torch.bfloat16)
        # trans_a=False over a .t() view folds to a transposed layout tag.
        prob = problem_of(a, b, trans_a=False, trans_b=False)
        assert prob.crosswise == 2  # folded A (crosswise) + non-transposed B

    @pytest.mark.parametrize(
        ("dt_a", "dt_b", "want"),
        [
            (torch.bfloat16, torch.bfloat16, 0),
            (torch.bfloat16, torch.int8, 1),
            (torch.bfloat16, torch.float8_e4m3fn, 1),
            (torch.int8, torch.int8, 2),
            (torch.float8_e4m3fn, torch.float8_e4m3fn, 3),
            (torch.float8_e5m2, torch.float8_e5m2, 3),
        ],
    )
    def test_perf_class_pairs(self, dt_a, dt_b, want):
        assert perf_class_of(dt_a, dt_b) == want

    def test_unsupported_pair_raises(self):
        with pytest.raises(ValueError):
            perf_class_of(torch.float32, torch.bfloat16)


class TestHeuristicRows:
    def test_covers_crosswise_only_and_shadows_nothing(self):
        rows = heuristic_rows(FACTS)
        assert len(rows) == 4 * 2 * 3  # classes x crosswise x M bands
        assert all(r.crosswise in (1, 2) for r in rows)

    def test_ring_feasible_and_demotes(self):
        rows = heuristic_rows(FACTS)
        for r in rows:
            bm, bn = CTA_GEOMETRY[r.cta]
            ba, bb = {0: (2, 2), 1: (2, 1), 2: (1, 1), 3: (1, 1)}[r.perf_class]
            assert ring_bytes(r.cta, r.stages, r.kk, ba, bb) <= FACTS["smem_max"]
        # A part that cannot opt in past 48KB keeps every ring inside it:
        # class 0's s3 small ring (64KB) demotes to s2, while the thinner
        # class 1 ring (exactly 48KB) legitimately keeps its depth.
        small = heuristic_rows({**FACTS, "smem_max": 48 * 1024})
        widths = {0: (2, 2), 1: (2, 1), 2: (1, 1), 3: (1, 1)}
        for r in small:
            ba, bb = widths[r.perf_class]
            assert ring_bytes(r.cta, r.stages, r.kk, ba, bb) <= 48 * 1024
        assert all(r.stages == 2 for r in small if r.perf_class == 0)

    def test_row_text_is_row_file_syntax(self):
        row = Row(63, 64, 127, 128, 0, 1, 2, 3, 32)
        assert row.text() == "63 64 127 128 0 1 2 3 0 32"


class TestTuneFlow:
    def _tuner(self, fake, monkeypatch_measure=None):
        tuner = GemmAutotuner()
        tuner._mod = fake
        if monkeypatch_measure is not None:
            tuner._measure = monkeypatch_measure
        assert tuner.start(time_budget_s=0)
        return tuner

    def test_start_installs_heuristic_floor(self):
        fake = FakeGemm()
        self._tuner(fake)
        assert len(fake.installs) == 1
        assert fake.installs[0].count("\n") == 4 * 2 * 3 - 1  # 24 rows

    def test_builtin_shapes_never_tune(self):
        fake = FakeGemm(source="builtin")
        tuner = self._tuner(fake)
        a, b = nt_operands()
        tuner.note(a, b, None, None, False, True, None)
        tuner.note(a, b, None, None, False, True, None)
        assert fake.probes == 1  # one probe per distinct shape
        assert len(fake.installs) == 1  # start() only: no candidates forced

    def test_degraded_shape_tunes_and_persists(self, tmp_path):
        fake = FakeGemm(source="degraded")
        measured: list = []

        def fake_measure(prob, candidates, *call_args):
            measured.extend(candidates)
            return candidates[-1]  # the "winner": last candidate

        tuner = self._tuner(fake, fake_measure)
        a, b = nt_operands()  # NT is crosswise 0, but the fake says degraded
        tuner.note(a, b, None, None, False, True, None)
        assert measured, "the degraded shape ran a candidate sweep"
        # Candidates were the cw-0 two-byte recipes that fit the smem.
        assert all(c.crosswise == 0 for c in measured)
        assert all(c.perf_class == 0 for c in measured)
        # start() + merged install = 2 installs.
        assert len(fake.installs) == 2
        # The winner merged in front of the heuristic floor.
        winner = tuner._rows[0]
        assert winner.recipe() == measured[-1].recipe()
        assert winner.m_min == 63 and winner.m_max == 64 + 64 // 4  # grown
        # Persisted under the device signature.
        cache = tmp_path / "cache" / f"{device_signature(FACTS)}.rows"
        assert cache.is_file()
        assert winner.text() in cache.read_text()

    def test_tuned_shape_does_not_retune(self):
        fake = FakeGemm(source="degraded")
        tuner = self._tuner(fake, lambda *a: None)
        a, b = nt_operands()
        tuner.note(a, b, None, None, False, True, None)
        installs_after_tune = len(fake.installs)
        # The refresh probe reports the injected tier now.
        fake.source = "injected"
        tuner.note(a, b, None, None, False, True, None)
        tuner.note(a, b, None, None, False, True, None)
        assert len(fake.installs) == installs_after_tune

    def test_env_table_idles_the_tuner(self, monkeypatch):
        monkeypatch.setenv("ASTR_GEMM_TABLE", "/nonexistent/rows.txt")
        fake = FakeGemm(source="degraded")
        tuner = self._tuner(fake, lambda *a: pytest.fail("must not measure"))
        a, b = nt_operands()
        tuner.note(a, b, None, None, False, True, None)
        assert len(fake.installs) == 0

    def test_max_shapes_cap(self):
        fake = FakeGemm(source="degraded")
        tuner = self._tuner(fake, lambda *a: None)
        tuner._tuned = tuner._max_shapes
        a, b = nt_operands()
        tuner.note(a, b, None, None, False, True, None)
        assert len(fake.installs) == 1  # start() only

    def test_same_recipe_winners_merge_bands(self):
        fake = FakeGemm(source="degraded")
        tuner = self._tuner(fake)
        first = Row(63, 64, 127, 128, 0, 0, 2, 3, 32)
        second = Row(199, 200, 127, 128, 0, 0, 2, 3, 32)
        tuner._merge(first)
        tuner._merge(second)
        assert len(tuner._rows) == 1
        merged = tuner._rows[0]
        assert merged.m_min == 63 and merged.m_max == 200 + 200 // 4

    def test_different_recipe_prepends(self):
        fake = FakeGemm(source="degraded")
        tuner = self._tuner(fake)
        tuner._merge(Row(63, 64, 127, 128, 0, 0, 2, 3, 32))
        tuner._merge(Row(63, 64, 127, 128, 0, 0, 0, 3, 64))
        assert len(tuner._rows) == 2
        # Newest measurement first: first match prefers the later winner.
        assert tuner._rows[0].cta == 0 and tuner._rows[1].cta == 2

    def test_cache_roundtrip(self, tmp_path):
        fake = FakeGemm(source="degraded")
        tuner = self._tuner(fake)
        tuner._merge(Row(63, 64, 127, 128, 0, 1, 2, 3, 32))
        tuner._persist()
        fresh = GemmAutotuner()
        fresh._mod = FakeGemm(source="degraded")
        assert fresh.start(time_budget_s=0)
        # The persisted row is the grown band _merge produced, and it comes
        # back byte-identical.
        assert tuner._rows[0] in fresh._rows
