"""Runtime plan-row autotuner unit tests.

The tune machinery is tested against a fake ``gemm`` module (the planner
bindings as call records), so these run without a GPU or a built
extension: what is under test is the hook's coverage logic (which shapes
tune, which never do), the candidate filter, the winner's interval
growth, and the per-device persistence roundtrip.
"""

import pytest
import torch

from astrai.extension.plan import (
    GemmAutotuner,
    Row,
    _problem_key,
    device_signature,
    heuristic_rows,
)

WIDTH_PERF = {(2, 2): 0, (2, 1): 1, (1, 1): 2}

FACTS = {
    "sms": 128,
    "smem_max": 101376,
    "smem_per_sm": 102400,
    "regs_per_sm": 65536,
    "l2_bytes": 75497472,
    "cc": 89,
}


# The cross-section of tile_vocabulary rows the fake serves, in the
# binding's 10-field form (cw, ba, bb, cta, stages, kk, bm, bn, threads,
# smem). smem follows policy.cuh's ring formula so the feasibility paths
# behave like the real vocabulary.
def _ring(bm, bn, kk, stages, ba, bb):
    return (stages + 1) * kk * (bm * ba + bn * bb)


_GEOMETRY = {0: (64, 64), 1: (128, 64), 2: (128, 128), 3: (128, 256), 4: (64, 128)}

VOCAB = [
    [
        cw,
        ba,
        bb,
        cta,
        stages,
        kk,
        *_GEOMETRY[cta],
        256,
        _ring(*_GEOMETRY[cta], kk, stages, ba, bb),
    ]
    for cw in (0, 1)
    for ba, bb in ((2, 2), (2, 1), (1, 1))
    for cta, stages, kk in (
        (0, 2, 64),
        (0, 3, 64),
        (1, 2, 64),
        (1, 3, 64),
        (2, 2, 64),
        (2, 3, 64),
        (0, 2, 32),
        (2, 2, 32),
    )
    if not (kk == 32 and (ba, bb) == (1, 1))  # kK=32 rides the congruous ladder
] + [
    [0, 1, 1, 3, 2, 64, *_GEOMETRY[3], 256, _ring(*_GEOMETRY[3], 64, 2, 1, 1)],
]


class FakeGemm:
    """The planner bindings as call records."""

    # The config-patch schema attribute the tuner's capability probe reads
    # (a stale build has ``configure`` too, so the name alone proves nothing).
    CONFIG_API = 2

    def __init__(self, source: str = "degraded", override_rows: int = 0):
        self.source = source
        self.override_rows = override_rows
        self.installs: list[str] = []
        self.probes = 0

    def config_state(self):
        return {
            "planner": "table",
            "planner_mode": 0,
            "log": False,
            "table": {
                "off": False,
                "override_rows": self.override_rows,
                "override_source": "",
                "injected_rows": 0,
                "injected_source": "",
            },
            "staging": {"tma": True, "mx": True},
        }

    _PERF = {
        (torch.bfloat16, torch.bfloat16): 0,
        (torch.bfloat16, torch.int8): 1,
        (torch.bfloat16, torch.float8_e4m3fn): 1,
        (torch.bfloat16, torch.float8_e5m2): 1,
        (torch.int8, torch.int8): 2,
        (torch.float8_e4m3fn, torch.float8_e4m3fn): 3,
        (torch.float8_e5m2, torch.float8_e5m2): 3,
    }

    def plan_probe(self, m, n, k, dt_a, dt_b, trans_a, trans_b, batch=1):
        self.probes += 1
        return {
            "source": self.source,
            "cta": 0,
            "stages": 2,
            "raster": 0,
            "kk": 64,
            "perf_class": self._PERF.get((dt_a, dt_b), 0),
            "crosswise": 0,
        }

    def configure(self, patch):
        # The tuner's only use of the config channel: rows at the injected
        # tier (below any user override). Anything else would be a bug here.
        assert patch["tier"] == "injected" and set(patch) == {"rows", "tier"}
        self.installs.append(patch["rows"])
        return self.config_state()

    def tile_vocabulary(self):
        return VOCAB

    def device_facts_info(self):
        return dict(FACTS)

    def quant_gemm(self, *args, **kwargs):
        return None


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("ASTR_GEMM_AUTOTUNE", raising=False)
    monkeypatch.setenv("ASTR_GEMM_TUNE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ASTR_GEMM_TUNE_TRIGGER", "1")
    monkeypatch.setenv("ASTR_GEMM_TUNE_MAX_SHAPES", "8")


def nt_operands(m=64, n=128, k=256):
    # trans_b=True takes B stored [N][K] (the weight / fused-linear form).
    a = torch.zeros(m, k, dtype=torch.bfloat16)
    b = torch.zeros(n, k, dtype=torch.bfloat16)
    return a, b


class TestProblemKey:
    def test_nt_production_route(self):
        a, b = nt_operands()
        m, n, k, batch, crosswise, dt_a, dt_b = _problem_key(a, b, False, True)
        assert (m, n, k, batch) == (64, 128, 256, 1)
        assert crosswise == 0  # dual-congruous, the fused-linear shape
        assert (dt_a, dt_b) == ("torch.bfloat16", "torch.bfloat16")

    def test_tt_is_crosswise(self):
        a = torch.zeros(256, 64, dtype=torch.bfloat16)  # [K][M]
        b = torch.zeros(128, 256, dtype=torch.bfloat16)  # [N][K]
        assert _problem_key(a, b, True, True)[4] == 1

    def test_tn_is_dual_crosswise(self):
        a = torch.zeros(256, 64, dtype=torch.bfloat16)  # [K][M]
        b = torch.zeros(256, 128, dtype=torch.bfloat16)  # [K][N]
        assert _problem_key(a, b, True, False)[4] == 2

    def test_col_major_view_folds_the_tag(self):
        base = torch.zeros(64, 256, dtype=torch.bfloat16)
        a = base.t()  # [256][64] storage, col-major view
        assert a.stride(-1) != 1 and a.stride(-2) == 1
        b = torch.zeros(128, 256, dtype=torch.bfloat16)
        # trans_a=False over a .t() view folds to a transposed layout tag.
        assert _problem_key(a, b, False, False)[4] == 2


class TestHeuristicRows:
    def test_covers_crosswise_only_and_shadows_nothing(self):
        rows = heuristic_rows(FACTS, VOCAB, WIDTH_PERF)
        pairs = len({(e[1], e[2]) for e in VOCAB})
        assert len(rows) == pairs * 2 * 3  # width pairs x crosswise x M bands
        assert all(r.crosswise in (1, 2) for r in rows)
        assert {r.perf_class for r in rows} == set(WIDTH_PERF.values())

    def test_ring_feasible_and_demotes(self):
        ring = {(e[3], e[4], e[5], e[1], e[2]): e[9] for e in VOCAB}
        widths_of_perf = {perf: widths for widths, perf in WIDTH_PERF.items()}

        def ring_of(r):
            ba, bb = widths_of_perf[r.perf_class]
            return ring[(r.cta, r.stages, r.kk, ba, bb)]

        for r in heuristic_rows(FACTS, VOCAB, WIDTH_PERF):
            assert ring_of(r) <= FACTS["smem_max"]
        # A part that cannot opt in past 48KB keeps every ring inside it:
        # the 2-byte s3 small ring (64KB) demotes to s2 while the thinner
        # (2, 1)-width ring (48KB) legitimately keeps its depth.
        small = heuristic_rows({**FACTS, "smem_max": 48 * 1024}, VOCAB, WIDTH_PERF)
        for r in small:
            assert ring_of(r) <= 48 * 1024
        assert all(
            r.stages == 2 for r in small if r.perf_class == 0
        )  # the 2-byte class demoted

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
        pairs = len({(e[1], e[2]) for e in VOCAB})
        assert fake.installs[0].count("\n") == pairs * 2 * 3 - 1

    def test_start_refuses_a_pre_patch_build(self):
        # A stale .so still has ``configure`` — it just takes keyword
        # arguments, which hasattr cannot see. The schema attr is the guard,
        # and refusing beats crashing inside the first install.
        fake = FakeGemm()
        fake.CONFIG_API = 1
        tuner = GemmAutotuner()
        tuner._mod = fake
        assert tuner.start(time_budget_s=0) is False
        assert fake.installs == []

    def test_builtin_shapes_never_tune(self):
        fake = FakeGemm(source="builtin")
        tuner = self._tuner(fake)
        a, b = nt_operands()
        seeded = fake.probes  # start() seeds one probe per supported pair
        tuner.note(a, b, None, None, False, True, None)
        tuner.note(a, b, None, None, False, True, None)
        assert fake.probes == seeded + 1  # one more probe per distinct shape
        assert len(fake.installs) == 1  # start() only: no candidates forced

    def test_model_answers_are_coverage(self):
        # The analytical planner serving a shape is ownership too: the
        # tuner never sweeps what the model already plans.
        fake = FakeGemm(source="model")
        tuner = self._tuner(fake, lambda *a: pytest.fail("must not measure"))
        a, b = nt_operands()
        tuner.note(a, b, None, None, False, True, None)
        assert len(fake.installs) == 1

    def test_degraded_shape_tunes_and_persists(self, tmp_path):
        fake = FakeGemm(source="degraded")
        measured: list = []

        def fake_measure(key, candidates, *call_args):
            measured.extend(candidates)
            return candidates[-1]  # the "winner": last candidate

        tuner = self._tuner(fake, fake_measure)
        a, b = nt_operands()  # NT is crosswise 0, but the fake says degraded
        tuner.note(a, b, None, None, False, True, None)
        assert measured, "the degraded shape ran a candidate sweep"
        # Candidates were the cw-0 two-byte recipes that fit the smem.
        assert all(c.crosswise == 0 for c in measured)
        assert all(c.perf_class == 0 for c in measured)
        assert all(c.kk in (32, 64) for c in measured)
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

    def test_override_table_idles_the_tuner(self):
        fake = FakeGemm(source="degraded", override_rows=3)
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
