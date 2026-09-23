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

from astrai.extension import ops, plan
from astrai.extension.loader import is_available

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not is_available("gemm"), reason="gemm kernel not built"),
]

SHAPE = (512, 11008, 4096)  # the wide-N band the analytical model wins
ROW = "511 513 8191 0 0 0 1 3 0 64"  # narrow CTA, 3 stages, kK 64


@pytest.fixture(autouse=True)
def _clean_plan_state():
    # Both row tiers, not just the override: set_table("") clears the
    # override rows only, so an injected row leaked from another test would
    # keep answering ahead of the planner under test.
    ops.gemm.set_table("")
    plan.configure(rows="", tier="injected")
    ops.gemm.set_planner("")  # back to the shipped default
    ops.gemm.set_staging()
    ops.gemm.set_log(False)
    yield
    ops.gemm.set_table("")
    plan.configure(rows="", tier="injected")
    ops.gemm.set_planner("")  # back to the shipped default
    ops.gemm.set_staging()
    ops.gemm.set_log(False)


class TestMode:
    def test_default_is_hybrid(self):
        # The shipped default: rows when any exist, else the model. The
        # compiled-in tables are empty, so a fresh process gets the model.
        state = ops.gemm.state()
        assert state["planner"] == "hybrid"
        assert ops.gemm.probe(*SHAPE)["source"] == "model"

    def test_model_mode_skips_the_rows(self):
        ops.gemm.set_planner("model")
        ops.gemm.set_table(ROW)
        assert ops.gemm.state()["planner"] == "model"
        assert ops.gemm.probe(*SHAPE)["source"] == "model"

    def test_hybrid_prefers_rows_then_model(self):
        ops.gemm.set_planner("hybrid")
        assert ops.gemm.probe(*SHAPE)["source"] == "model"
        ops.gemm.set_table(ROW)  # a row now owns the shape
        assert ops.gemm.probe(*SHAPE)["source"] == "override"
        ops.gemm.set_table("-")  # every row tier off
        assert ops.gemm.probe(*SHAPE)["source"] == "model"

    def test_model_only_ignores_the_table(self):
        ops.gemm.set_planner("model")
        ops.gemm.set_table(ROW)
        assert ops.gemm.probe(*SHAPE)["source"] == "model"

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            ops.gemm.set_planner("cost-model")


class TestTable:
    def test_override_rows_take_the_shape(self):
        installed = ops.gemm.set_table(ROW)
        assert installed == 1
        info = ops.gemm.probe(*SHAPE)
        assert info["source"] == "override"
        assert (info["cta"], info["stages"], info["kk"]) == (1, 3, 64)

    def test_off_mode_kills_the_rows_only(self):
        # "-" disables override, injected and builtin alike; what answers
        # after that is the planner mode's business: the model under the
        # shipped hybrid default, the degraded ladder under "table".
        ops.gemm.set_table("-")
        assert ops.gemm.probe(*SHAPE)["source"] == "model"
        ops.gemm.set_planner("table")
        assert ops.gemm.probe(*SHAPE)["source"] == "degraded"

    def test_clear_restores_the_default(self):
        ops.gemm.set_table(ROW)
        ops.gemm.set_table("")
        assert ops.gemm.state()["table"]["override_rows"] == 0
        # The builtin tables ship empty, so the model answers again.
        assert ops.gemm.probe(*SHAPE)["source"] == "model"

    def test_injected_rows_rank_below_override(self):
        plan.configure(rows=ROW, tier="injected")
        assert ops.gemm.state()["table"]["injected_rows"] == 1
        assert ops.gemm.probe(*SHAPE)["source"] == "injected"
        ops.gemm.set_table("511 513 8191 0 0 0 2 2 0 64")
        assert ops.gemm.probe(*SHAPE)["source"] == "override"


class TestStaging:
    def test_state_reports_the_switches(self):
        assert ops.gemm.state()["staging"] == {"tma": True, "mx": True}
        ops.gemm.set_staging(tma=False)
        assert ops.gemm.state()["staging"] == {"tma": False, "mx": True}
        ops.gemm.set_staging(mx=False)
        assert ops.gemm.state()["staging"] == {"tma": False, "mx": False}


class TestCompiledInTables:
    def test_rows_ship_device_guarded(self):
        # Compiled-in rows are the measured diff of the model's errors for
        # ONE device (the GENERATED block's provenance); the tier is
        # signature-guarded, so it serves exactly there. On any other part
        # "builtin" never appears, which is what keeps the rows from
        # leaking onto a machine they were not measured on.
        sig = ops.gemm.facts()
        measured_here = (
            sig["cc"] == 120
            and sig["sms"] == 170
            and sig["smem_per_sm"] == 102400
            and sig["l2_bytes"] == 100663296
        )
        in_band = ((128, 2048, 4096), (2048, 14336, 4096))
        for shape in in_band:
            src = ops.gemm.probe(*shape)["source"]
            if measured_here:
                assert src == "builtin"
            else:
                assert src != "builtin"
        # Bands the model already wins stay the model's even where the
        # rows are live (the diff only claims measured >=2% wins).
        assert ops.gemm.probe(512, 11008, 4096)["source"] != "override"


class TestProbe:
    def test_reports_the_query_key(self):
        info = ops.gemm.probe(*SHAPE)
        assert info["perf_class"] == 0  # bf16 x bf16
        assert info["crosswise"] == 0  # the NT fused-linear shape

    def test_vocabulary_carries_geometry(self):
        rows = ops.gemm.tile_vocabulary()
        assert rows, "the vocabulary must not be empty"
        for entry in rows:
            (
                crosswise,
                ba,
                bb,
                cta,
                stages,
                kk,
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
            assert stages in (2, 3)
            assert kk in (32, 64)
            assert bm in (64, 128) and bn in (64, 128, 256)
            # the warp tiling spells the recipe name's W<x>x<y>, and the
            # threads count follows it ((bm/wm)*(bn/wn)*32)
            assert (bm // wm) * (bn // wn) * 32 == threads
            assert threads > 0 and smem > 0

    def test_facts_are_populated(self):
        facts = ops.gemm.facts()
        assert facts["sms"] > 0
        assert facts["cc"] >= 80
        assert facts["l2_bytes"] > 0


class TestModelRule:
    """The planner's model rule, asserted as a RULE rather than as a
    recipe so it holds on any device.

    Two-byte pairs rank on the cost alone (2026-09-16 full-grid re-fit):
    the (resident, stages) prefix measured a net loss there — the stages
    tier preferred the S3 twin in the cells where S2 measured faster — and
    kK now has a term of its own, so what residency was right about is
    recovered by the cost. The cost is (operand + output + issue) bytes
    times waves times resident, where the issue term prices every
    k-iteration at a fixed charge per accumulator cell (a deeper kK
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

    # gemm.cuh kKTileIssueBytes / kMmaArmBytesPerInstr — keep in sync
    # (both fitted on the 2026-09-16 grid, RTX 5090, bf16 out).
    K_TILE_ISSUE_BYTES = 8
    MMA_ARM_BYTES_PER_INSTR = 64

    @classmethod
    def _cost(cls, entry, m, n, k, facts):
        """cost_of mirrored from gemm.cuh, or None when the ring is not
        resident on this device at all — the only way to assert the rule
        from Python. Both staging forms: the planner branches on the
        device's cc (TMA needs sm_90+), so on an sm_89 part (L20/4090)
        the zero-constant cp.async form is the one under test.
        """
        _cw, _ba, _bb, _cta, _stages, kk, bm, bn, _wm, _wn, _threads, smem = entry
        resident = min(facts["smem_per_sm"] // smem, 2 if smem <= 48 * 1024 else 1)
        if resident <= 0:
            return None
        blocks = ((m + bm - 1) // bm) * ((n + bn - 1) // bn)
        output = 2 * bm * bn
        if facts["cc"] < 90:  # cp.async staging: raw-floor residency divides
            operand = ((k + kk - 1) // kk) * kk * (bm * 2 + bn * 2)
            mu = facts["smem_per_sm"] // smem
            slots = facts["sms"] * mu
            waves = (blocks + slots - 1) // slots if slots > 0 else 1
            return (operand + output) * waves
        operand = k * (bm * 2 + bn * 2)
        issue = cls.K_TILE_ISSUE_BYTES * bm * bn * ((k + kk - 1) // kk)
        mma_arm = cls.MMA_ARM_BYTES_PER_INSTR * bm * bn * k // (128 * 16)
        per_cta = max(operand + output + issue, mma_arm)
        slots = facts["sms"] * resident
        waves = (blocks + slots - 1) // slots if slots > 0 else 1
        return per_cta * waves * resident

    def test_two_byte_pick_attains_the_minimum_cost(self):
        # The rule under test is the analytical model's own; pin the
        # planner to it so a compiled-in row (which outranks the model in
        # the default chain) cannot answer in its place. Staging is pinned
        # on too: the cost branches on it, and a leaked tma=False from an
        # earlier test would silently move the pick to the cp.async form.
        ops.gemm.set_planner("model")
        ops.gemm.set_staging(tma=True)
        facts = ops.gemm.facts()
        vocab = [
            entry
            for entry in ops.gemm.tile_vocabulary()
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

            info = ops.gemm.probe(*shape)
            assert info["source"] == "model", shape
            picked = (info["cta"], info["stages"], info["kk"])
            assert picked in by_recipe, f"{shape}: {picked} is not a candidate"
            assert by_recipe[picked] == min(by_recipe.values()), (
                f"{shape}: picked {picked} with cost {by_recipe[picked]}, "
                f"the best is {min(by_recipe.values())}"
            )


class TestPlanFacade:
    """The ``plan`` value API over the same bindings the flat names call."""

    def test_config_agrees_with_the_wire(self):
        cfg = plan.config
        wire = ops.gemm.state()
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
        assert after.planner == "hybrid"  # unset resolves to the shipped default

    def test_override_restores_knobs_and_rows(self):
        ops.gemm.set_table(ROW)  # a tier the block must bring back
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
        ops.gemm.set_table(ROW)
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
        # The injected tier sits below the override one, so the same row
        # reports a different source depending on which is installed.
        plan.configure(rows=ROW, tier="injected")
        assert plan.config.injected_rows == 1
        assert plan.config.override_rows == 0
        assert ops.gemm.probe(*SHAPE)["source"] == "injected"
        plan.configure(rows=ROW, tier="override")
        assert ops.gemm.probe(*SHAPE)["source"] == "override"

    def test_probe_facts_tiles_are_records(self):
        info = plan.probe(*SHAPE)
        assert info.source == info["source"]
        assert isinstance(info.cta, int)
        assert plan.facts.cc == ops.gemm.facts()["cc"]
        tiles = plan.tiles()
        raw = ops.gemm.tile_vocabulary()
        assert len(tiles) == len(raw)
        first = tiles[0]
        assert first[3] == first.cta  # row[3] still the CTA class
        crosswise, ba, bb, cta, stages, kk, bm, bn, wm, wn, threads, smem = first
        assert (crosswise, ba, bb, cta) == tuple(raw[0][:4])
        assert first.cta_name and first.name.startswith("Tile_")
        assert first.name.endswith(f"_S{stages}")


class TestCrossLanguageSpellings:
    """The two formats Python writes and C++ reads: the bench's ``Tile_...``
    names and the plan-row text. Both are pinned here, so a drift fails in
    the suite instead of surfacing as a bad join in a sweep."""

    # The reader the tools run over the bench's names (tune_plan_table).
    _NAME_RE = re.compile(r"Tile_(\d+)x(\d+)x(\d+)_W(\d+)x(\d+)_S(\d+)")

    def test_tile_name_spells_the_record(self):
        for tile in plan.tiles():
            assert tile.name == (
                f"Tile_{tile.bm}x{tile.bn}x{tile.kk}"
                f"_W{tile.wm}x{tile.wn}_S{tile.stages}"
            ), tile

    def test_tile_name_reads_back_as_its_recipe(self):
        for tile in plan.tiles():
            m = self._NAME_RE.fullmatch(tile.name)
            assert m, tile.name
            assert tuple(int(part) for part in m.groups()) == (
                tile.bm,
                tile.bn,
                tile.kk,
                tile.wm,
                tile.wn,
                tile.stages,
            ), tile

    def test_row_text_means_the_same_to_the_planner(self):
        # A row spelled in Python must make the C++ planner pick exactly that
        # recipe: install it, ask who serves the band, and check the answer
        # echoes the numbers the text carried (the record keeps the spelling
        # in step with the vocabulary, so the row is legal by construction).
        tile = next(t for t in plan.tiles() if (t.crosswise, t.ba, t.bb) == (0, 2, 2))
        ops.gemm.set_table(f"511 513 8191 0 0 0 {tile.cta} {tile.stages} 0 {tile.kk}")
        picked = plan.probe(*SHAPE)
        assert picked.source == "override"
        assert (picked.cta, picked.stages, picked.kk) == (
            tile.cta,
            tile.stages,
            tile.kk,
        )
