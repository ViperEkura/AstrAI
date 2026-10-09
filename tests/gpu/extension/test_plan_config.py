"""Runtime plan API tests (the gemm adapter's set_*/probe surface).

These exercise the C++ planner through the real binding, so they need a
built gemm module and a CUDA device. The configuration API's contract —
precedence, modes, table install/clear, the probe report — is what the
first classes cover; the analytical planner's own selection RULE is
covered by TestModelRule below.
"""

import re

import pytest
import torch

from astrai.extension import kernel, plan
from astrai.extension.runtime.loader import is_available

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not is_available("gemm"), reason="gemm kernel not built"),
]

SHAPE = (512, 11008, 4096)  # the wide-N band the analytical model wins
ROW = "511 513 8191 0 0 0 1 3 0 64"  # narrow CTA, 3 pipeline stages, k_tile 64


@pytest.fixture(autouse=True)
def _clean_plan_state():
    # Both row tiers, not just the override: set_table("") clears the
    # override rows only, so an injected row leaked from another test would
    # keep answering ahead of the planner under test.
    kernel.gemm.set_table("")
    plan.configure(rows="", tier="injected")
    kernel.gemm.set_planner("")  # back to the shipped default
    kernel.gemm.set_staging()
    kernel.gemm.set_log(False)
    yield
    kernel.gemm.set_table("")
    plan.configure(rows="", tier="injected")
    kernel.gemm.set_planner("")  # back to the shipped default
    kernel.gemm.set_staging()
    kernel.gemm.set_log(False)


class TestMode:
    def test_default_is_model(self):
        # The shipped default uses the legacy model directly.
        state = kernel.gemm.state()
        assert state["planner"] == "model"
        assert kernel.gemm.probe(*SHAPE)["source"] == "model"
        default = kernel.gemm.probe(*SHAPE)
        with plan.override(planner="model"):
            explicit = kernel.gemm.probe(*SHAPE)
        assert (default["cta"], default["k_stages"], default["k_tile"]) == (
            explicit["cta"],
            explicit["k_stages"],
            explicit["k_tile"],
        )

    def test_model_mode_skips_the_rows(self):
        kernel.gemm.set_planner("model")
        kernel.gemm.set_table(ROW)
        assert kernel.gemm.state()["planner"] == "model"
        assert kernel.gemm.probe(*SHAPE)["source"] == "model"

    def test_hybrid_prefers_rows_then_heuristic(self):
        kernel.gemm.set_planner("hybrid")
        assert kernel.gemm.probe(*SHAPE)["source"] == "heuristic"
        kernel.gemm.set_table(ROW)  # a row now owns the shape
        assert kernel.gemm.probe(*SHAPE)["source"] == "override"
        kernel.gemm.set_table("-")  # every row tier off
        assert kernel.gemm.probe(*SHAPE)["source"] == "heuristic"

    def test_model_only_ignores_the_table(self):
        kernel.gemm.set_planner("model")
        kernel.gemm.set_table(ROW)
        assert kernel.gemm.probe(*SHAPE)["source"] == "model"

    def test_heuristic_skips_rows_and_does_not_reuse_model_cache(self):
        kernel.gemm.set_table(ROW)
        plan.configure(rows=ROW, tier="injected")
        for mode in ("model", "heuristic", "model", "heuristic"):
            with plan.override(planner=mode):
                assert kernel.gemm.state()["planner"] == mode
                assert kernel.gemm.probe(*SHAPE)["source"] == mode

    @pytest.mark.parametrize("mode", ["hybrid", "table", "heuristic", "model"])
    @pytest.mark.parametrize(
        "shape",
        [
            (0, 64, 32),
            (8, 0, 32),
            (8, 64, 0),
            (-1, 64, 32),
            (8, -1, 32),
            (8, 64, -1),
        ],
    )
    def test_probe_requires_positive_dimensions(self, mode, shape):
        with plan.override(planner=mode):
            with pytest.raises(
                RuntimeError, match="M, N and K must be greater than zero"
            ):
                kernel.gemm.probe(*shape)

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            kernel.gemm.set_planner("cost-model")


class TestTable:
    def test_override_rows_take_the_shape(self):
        kernel.gemm.set_planner("hybrid")
        installed = kernel.gemm.set_table(ROW)
        assert installed == 1
        info = kernel.gemm.probe(*SHAPE)
        assert info["source"] == "override"
        assert (info["cta"], info["k_stages"], info["k_tile"]) == (1, 3, 64)

    def test_off_mode_kills_the_rows_only(self):
        kernel.gemm.set_planner("hybrid")
        # "-" disables override, injected and builtin alike; what answers
        # after that is the planner mode's business: the heuristic under the
        # shipped hybrid default, a missing-row error under "table".
        kernel.gemm.set_table("-")
        assert kernel.gemm.probe(*SHAPE)["source"] == "heuristic"
        kernel.gemm.set_planner("table")
        with pytest.raises(RuntimeError, match="no eligible recipe"):
            kernel.gemm.probe(*SHAPE)

    def test_table_miss_rejects_instead_of_guessing(self):
        with plan.override(planner="table", rows=ROW, tier="override"):
            with pytest.raises(RuntimeError, match="no eligible recipe"):
                kernel.gemm.probe(128, 1024, 256, trans_a=True)

    def test_clear_restores_the_default(self):
        kernel.gemm.set_planner("hybrid")
        kernel.gemm.set_table(ROW)
        kernel.gemm.set_table("")
        assert kernel.gemm.state()["table"]["override_rows"] == 0
        # This shape has no builtin row, so the heuristic answers again.
        assert kernel.gemm.probe(*SHAPE)["source"] == "heuristic"

    def test_injected_rows_rank_below_override(self):
        kernel.gemm.set_planner("hybrid")
        plan.configure(rows=ROW, tier="injected")
        assert kernel.gemm.state()["table"]["injected_rows"] == 1
        assert kernel.gemm.probe(*SHAPE)["source"] == "injected"
        kernel.gemm.set_table("511 513 8191 0 0 0 2 2 0 64")
        assert kernel.gemm.probe(*SHAPE)["source"] == "override"


class TestStaging:
    def test_state_reports_the_switches(self):
        assert kernel.gemm.state()["staging"] == {"tma": True, "mx": True}
        kernel.gemm.set_staging(tma=False)
        assert kernel.gemm.state()["staging"] == {"tma": False, "mx": True}
        kernel.gemm.set_staging(mx=False)
        assert kernel.gemm.state()["staging"] == {"tma": False, "mx": False}


class TestCompiledInTables:
    def test_rows_ship_device_guarded(self):
        kernel.gemm.set_planner("hybrid")
        # Compiled-in rows are the measured diff of the model's errors for
        # ONE device (the GENERATED block's provenance); the tier is
        # signature-guarded, so it serves exactly there. On any other part
        # "builtin" never appears, which is what keeps the rows from
        # leaking onto a machine they were not measured on.
        sig = kernel.gemm.facts()
        measured_here = (
            sig["cc"] == 120
            and sig["sms"] == 170
            and sig["smem_per_sm"] == 102400
            and sig["l2_bytes"] == 100663296
        )
        in_band = ((128, 2048, 4096), (2048, 14336, 4096))
        for shape in in_band:
            src = kernel.gemm.probe(*shape)["source"]
            if measured_here:
                assert src == "builtin"
            else:
                assert src != "builtin"
        if measured_here:
            tuned = kernel.gemm.probe(768, 6144, 1536, torch.int8, torch.int8)
            assert (tuned["source"], tuned["k_tile"]) == ("builtin", 128)
            outside = kernel.gemm.probe(1536, 6144, 1536, torch.int8, torch.int8)
            assert outside["k_tile"] != 128
        # Bands the model already wins stay the model's even where the
        # rows are live (the diff only claims measured >=2% wins).
        assert kernel.gemm.probe(512, 11008, 4096)["source"] != "override"


class TestProbe:
    def test_k_tile_and_k_stages_names(self):
        raw = kernel.gemm.probe(*SHAPE)
        assert raw["k_tile"] in (32, 64, 128)
        assert raw["k_stages"] in (2, 3)
        assert "kk" not in raw and "stages" not in raw

        decision = plan.probe(*SHAPE)
        assert decision.k_tile == raw["k_tile"]
        assert decision.k_stages == raw["k_stages"]

        tile = plan.tiles()[0]
        assert tile.k_tile in (32, 64, 128)
        assert tile.k_stages in (2, 3)

    def test_reports_the_query_key(self):
        info = kernel.gemm.probe(*SHAPE)
        assert info["perf_class"] == 0  # bf16 x bf16
        assert info["crosswise"] == 0  # the NT fused-linear shape

    def test_vocabulary_carries_geometry(self):
        rows = kernel.gemm.tile_vocabulary()
        assert rows, "the vocabulary must not be empty"
        for entry in rows:
            (
                crosswise,
                ba,
                bb,
                cta,
                k_stages,
                k_tile,
                bm,
                bn,
                wm,
                wn,
                threads,
                smem,
            ) = entry
            assert (crosswise, ba, bb) in (
                (0, 2, 2),
                (1, 2, 2),
                (0, 2, 1),
                (1, 2, 1),
                (0, 1, 1),
                (1, 1, 1),
            )
            assert cta in (0, 1, 2, 3, 4)
            assert k_stages in (2, 3)
            assert k_tile in (32, 64, 128)
            assert bm in (64, 128) and bn in (64, 128, 256)
            # the warp tiling spells the recipe name's W<x>x<y>, and the
            # threads count follows it ((bm/wm)*(bn/wn)*32)
            assert (bm // wm) * (bn // wn) * 32 == threads
            assert threads > 0 and smem > 0

    def test_facts_are_populated(self):
        facts = kernel.gemm.facts()
        assert facts["sms"] > 0
        assert facts["cc"] >= 80
        assert facts["l2_bytes"] > 0


class TestModelRule:
    """The planner's model rule, asserted as a RULE rather than as a
    recipe so it holds on any device.

    Two-byte pairs rank on the cost alone (2026-09-16 full-grid re-fit):
    the (resident, k_stages) prefix measured a net loss there — the k_stages
    tier preferred the S3 twin in the cells where S2 measured faster — and
    kK now has a term of its own, so what residency was right about is
    recovered by the cost. The cost is (operand + output + mainloop) equivalent bytes
    times waves times resident. The fitted mainloop term prices each
    k-iteration per accumulator cell (a deeper kK
    amortises it). This pins the planner to that cost and checks the pick
    attains its minimum. The mixed-pair residency gate and the byte-pair
    form are covered by model_capture's --check fidelity gate, which walks
    whole datasets.
    """

    SHAPES = (
        (512, 11008, 4096),
        (4096, 1536, 1536),
        (4096, 4096, 4096),
        (4096, 11008, 4096),
        (2048, 28672, 8192),
    )

    # planning.cpp kMainloopBytesPerCellTile / kMmaArmBytesPerInstr
    # (both fitted on the 2026-09-16 grid, RTX 5090, bf16 out).
    MAINLOOP_BYTES_PER_CELL_TILE = 8
    MMA_ARM_BYTES_PER_INSTR = 64

    @classmethod
    def _cost(cls, entry, m, n, k, facts):
        """cost_of mirrored from planning.cpp, or None when the ring is not
        resident on this device at all — the only way to assert the rule
        from Python. Both staging forms: the planner branches on the
        device's cc (TMA needs sm_90+), so on an sm_89 part (L20/4090)
        the zero-constant cp.async form is the one under test.
        """
        _cw, _ba, _bb, _cta, _k_stages, k_tile, bm, bn, _wm, _wn, _threads, smem = entry
        tma = kernel.gemm.capabilities()["tma"]
        staged = smem + 1024 + 16 * (_k_stages + 1) if tma else smem
        resident = min(facts["smem_per_sm"] // staged, 2 if smem <= 48 * 1024 else 1)
        if resident <= 0 or staged > facts["smem_max"]:
            return None
        blocks = ((m + bm - 1) // bm) * ((n + bn - 1) // bn)
        output = 2 * bm * bn
        if not tma:  # effective compiled staging
            operand = ((k + k_tile - 1) // k_tile) * k_tile * (bm * 2 + bn * 2)
            mu = facts["smem_per_sm"] // smem
            slots = facts["sms"] * mu
            waves = (blocks + slots - 1) // slots if slots > 0 else 1
            return (operand + output) * waves
        operand = k * (bm * 2 + bn * 2)
        loop_penalty = (
            cls.MAINLOOP_BYTES_PER_CELL_TILE * bm * bn * ((k + k_tile - 1) // k_tile)
        )
        mma_instructions = (
            ((k + k_tile - 1) // k_tile) * (k_tile // 16) * (bm // 16) * (bn // 8)
        )
        mma_arm = cls.MMA_ARM_BYTES_PER_INSTR * mma_instructions
        per_cta = max(operand + output + loop_penalty, mma_arm)
        slots = facts["sms"] * resident
        waves = (blocks + slots - 1) // slots if slots > 0 else 1
        return per_cta * waves * resident

    def test_two_byte_pick_attains_the_minimum_cost(self):
        # The rule under test is the analytical model's own; pin the
        # planner to it so a compiled-in row (which outranks the model in
        # the default chain) cannot answer in its place. Staging is pinned
        # on too: the cost branches on it, and a leaked tma=False from an
        # earlier test would silently move the pick to the cp.async form.
        kernel.gemm.set_planner("model")
        kernel.gemm.set_staging(tma=True)
        facts = kernel.gemm.facts()
        vocab = [
            entry
            for entry in kernel.gemm.tile_vocabulary()
            if (entry[0], entry[1], entry[2]) == (0, 2, 2)  # NT, bf16 x bf16
        ]
        assert vocab, "the vocabulary carries no bf16 x bf16 candidates"
        for shape in self.SHAPES:
            m, n, k = shape
            by_recipe = {}
            for entry in vocab:
                cost = self._cost(entry, m, n, k, facts)
                if cost is not None:
                    by_recipe[tuple(entry[3:6])] = cost
            assert by_recipe, f"no resident candidate for {shape}"

            info = kernel.gemm.probe(*shape)
            assert info["source"] == "model", shape
            picked = (info["cta"], info["k_stages"], info["k_tile"])
            assert picked in by_recipe, f"{shape}: {picked} is not a candidate"
            assert by_recipe[picked] == min(by_recipe.values()), (
                f"{shape}: picked {picked} with cost {by_recipe[picked]}, "
                f"the best is {min(by_recipe.values())}"
            )


class TestHeuristicLaunch:
    """Exercise actual selected kernels, including tails and layout rewrites."""

    @pytest.mark.parametrize(
        "pair",
        [
            (torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, torch.int8),
            (torch.int8, torch.int8),
            (torch.float8_e4m3fn, torch.float8_e4m3fn),
            (torch.float8_e5m2, torch.float8_e5m2),
        ],
    )
    @pytest.mark.parametrize("tma", [False, True])
    @pytest.mark.parametrize("batch", [1, 3])
    @pytest.mark.parametrize(
        "trans_a,trans_b", [(False, True), (True, True), (False, False), (True, False)]
    )
    def test_tail_batch_and_layout_correctness(
        self, pair, tma, batch, trans_a, trans_b
    ):
        if (
            pair[0] in (torch.float8_e4m3fn, torch.float8_e5m2)
            and not kernel.gemm.capabilities()["fp8"]
        ):
            pytest.skip("FP8 kernels unavailable")
        m, n, k = 33, 67, 80
        prefix = () if batch == 1 else (batch,)
        a_math = torch.randint(-2, 3, (*prefix, m, k), device="cuda").float()
        b_math = torch.randint(-2, 3, (*prefix, k, n), device="cuda").float()
        a = (a_math.transpose(-2, -1) if trans_a else a_math).to(pair[0]).contiguous()
        b = (b_math.transpose(-2, -1) if trans_b else b_math).to(pair[1]).contiguous()
        sa = None if pair[0] == torch.bfloat16 else torch.tensor([0.5], device="cuda")
        sb = None if pair[1] == torch.bfloat16 else torch.tensor([0.25], device="cuda")
        expected = a_math @ b_math
        if sa is not None:
            expected *= sa
        if sb is not None:
            expected *= sb
        with plan.override(planner="heuristic", tma=tma, mx=False):
            actual = kernel.gemm.quant_gemm(a, b, sa, sb, trans_a, trans_b)
            info = kernel.gemm.probe(
                m, n, k, *pair, trans_a=trans_a, trans_b=trans_b, batch=batch
            )
        assert info["source"] == "heuristic"
        torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0)


class TestPlanFacade:
    """The ``plan`` value API over the same bindings the flat names call."""

    def test_config_agrees_with_the_wire(self):
        cfg = plan.config
        wire = kernel.gemm.state()
        assert cfg.planner == wire["planner"]
        assert cfg.planner_mode == wire["planner_mode"]
        assert cfg.log == wire["log"]
        assert cfg.table_off == wire["table"]["off"]
        assert cfg.override_rows == wire["table"]["override_rows"]
        assert cfg.override_source == wire["table"]["override_source"]
        assert cfg.staging.tma == wire["staging"]["tma"]  # the sub-record
        assert cfg["staging"]["mx"] == wire["staging"]["mx"]  # still a mapping

    def test_configure_returns_the_new_value(self):
        after = plan.configure(planner="model", log=True)
        assert after.planner == "model" and after.log is True
        assert plan.config.planner == "model"

    def test_configure_rejects_unknown_spellings(self):
        with pytest.raises(ValueError):
            plan.configure(planner="nope")
        with pytest.raises(ValueError):
            plan.configure(rows=ROW, tier="middle")

    def test_planner_reset_does_not_skip_the_rest_of_the_patch(self):
        # A planner reset ("" -> unset) applies the rest of the patch too:
        # the early return the old binding had is gone, which is what makes
        # a saved config fully re-installable in one call.
        plan.configure(planner="model", log=True)
        after = plan.configure(planner="", log=False)
        assert after.planner_mode == -1 and after.log is False
        assert after.planner == "model"  # unset resolves to the shipped default

    def test_override_restores_knobs_and_rows(self):
        kernel.gemm.set_table(ROW)  # a tier the block must bring back
        plan.configure(planner="model", tma=False)
        before = plan.config
        with plan.override(
            planner="table", tma=True, rows="", tier="override"
        ) as inside:
            assert inside.planner == "table" and inside.staging.tma is True
            assert inside.override_rows == 0
        after = plan.config
        assert after == before, "the block must restore the exact prior state"

    def test_override_restores_on_exception(self):
        before = plan.config
        with pytest.raises(RuntimeError):
            with plan.override(planner="table", mx=False):
                raise RuntimeError("boom")
        assert plan.config == before

    def test_saved_config_reinstalls_in_one_call(self):
        kernel.gemm.set_table(ROW)
        plan.configure(planner="table", log=True, tma=False)
        saved = plan.config
        plan.configure(planner="model", log=False, tma=True, rows="", tier="override")
        restored = plan.configure(
            planner=saved.planner_mode,
            log=saved.log,
            tma=saved.staging.tma,
            mx=saved.staging.mx,
            table_off=saved.table_off,
            rows=saved.override_source,
            tier="override",
        )
        assert restored == saved

    def test_rows_address_the_tier(self):
        plan.configure(planner="hybrid")
        # The injected tier sits below the override one, so the same row
        # reports a different source depending on which is installed.
        plan.configure(rows=ROW, tier="injected")
        assert plan.config.injected_rows == 1
        assert plan.config.override_rows == 0
        assert kernel.gemm.probe(*SHAPE)["source"] == "injected"
        plan.configure(rows=ROW, tier="override")
        assert kernel.gemm.probe(*SHAPE)["source"] == "override"

    def test_probe_facts_tiles_are_records(self):
        info = plan.probe(*SHAPE)
        assert info.source == info["source"]
        assert isinstance(info.cta, int)
        assert plan.facts.cc == kernel.gemm.facts()["cc"]
        tiles = plan.tiles()
        raw = kernel.gemm.tile_vocabulary()
        assert len(tiles) == len(raw)
        first = tiles[0]
        assert first[3] == first.cta  # row[3] still the CTA class
        crosswise, ba, bb, cta, k_stages, k_tile, bm, bn, wm, wn, threads, smem = first
        assert (crosswise, ba, bb, cta) == tuple(raw[0][:4])
        assert first.cta_name and first.name.startswith("Tile_")
        assert first.name.endswith(f"_S{k_stages}")


class TestCrossLanguageSpellings:
    """The two formats Python writes and C++ reads: the bench's ``Tile_...``
    names and the plan-row text. Both are pinned here, so a drift fails in
    the suite instead of surfacing as a bad join in a sweep."""

    # The reader the tools run over the bench's names (tune_plan_table).
    _NAME_RE = re.compile(r"Tile_(\d+)x(\d+)x(\d+)_W(\d+)x(\d+)_S(\d+)")

    def test_tile_name_spells_the_record(self):
        for tile in plan.tiles():
            assert tile.name == (
                f"Tile_{tile.bm}x{tile.bn}x{tile.k_tile}"
                f"_W{tile.wm}x{tile.wn}_S{tile.k_stages}"
            ), tile

    def test_tile_name_reads_back_as_its_recipe(self):
        for tile in plan.tiles():
            m = self._NAME_RE.fullmatch(tile.name)
            assert m, tile.name
            assert tuple(int(part) for part in m.groups()) == (
                tile.bm,
                tile.bn,
                tile.k_tile,
                tile.wm,
                tile.wn,
                tile.k_stages,
            ), tile

    def test_row_text_means_the_same_to_the_planner(self):
        kernel.gemm.set_planner("hybrid")
        # A row spelled in Python must make the C++ planner pick exactly that
        # recipe: install it, ask who serves the band, and check the answer
        # echoes the numbers the text carried (the record keeps the spelling
        # in step with the vocabulary, so the row is legal by construction).
        tile = next(t for t in plan.tiles() if (t.crosswise, t.ba, t.bb) == (0, 2, 2))
        kernel.gemm.set_table(
            f"511 513 8191 0 0 0 {tile.cta} {tile.k_stages} 0 {tile.k_tile}"
        )
        picked = plan.probe(*SHAPE)
        assert picked.source == "override"
        assert (picked.cta, picked.k_stages, picked.k_tile) == (
            tile.cta,
            tile.k_stages,
            tile.k_tile,
        )


@pytest.mark.parametrize("tma", [False, True])
def test_heuristic_resources_match_launched_geometry(tma):
    with plan.override(planner="heuristic", tma=tma, mx=False):
        info = plan.probe(512, 4096, 4096)
    resources = {row[:3]: row for row in info.resources}
    assert resources
    for row in resources.values():
        (
            _cta,
            _k_stages,
            _k_tile,
            bm,
            bn,
            k_tile,
            wm,
            wn,
            threads,
            resident,
            regs,
            local,
        ) = row
        assert threads == (bm // wm) * (bn // wn) * 32
        assert resident >= 0 and regs >= 0 and local >= 0
        if resident:
            assert regs > 0
    # The named 64x64x64 recipe is actually widened to sixteen warps.
    small = resources[(0, 2, 64)]
    assert small[6:9] == (16, 16, 512)
    selected = resources[(info.cta, info.k_stages, info.k_tile)]
    assert selected[9] > 0


@pytest.mark.parametrize("k", [79, 80])
def test_heuristic_probe_uses_contiguous_operand_alignment(k):
    with plan.override(planner="heuristic", tma=True, mx=False):
        info = plan.probe(33, 67, k)
        if k == 79:
            assert not info.tma
        a = torch.randint(-2, 3, (33, k), device="cuda").bfloat16()
        b = torch.randint(-2, 3, (67, k), device="cuda").bfloat16()
        actual = kernel.gemm.quant_gemm(a, b)
        torch.testing.assert_close(
            actual, (a.float() @ b.float().T).bfloat16(), rtol=0, atol=0
        )
