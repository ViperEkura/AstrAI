"""The declarative table behind strategy-owned checkpoint extras.

``optional_extras.COMPONENT_EXTRAS`` is the single source for which
component states ride in a checkpoint, where they live on the strategy, and
what a missing entry means on resume.  Saving walks the table; each component
restores its own entry through :func:`load_component_extra` (build order
differs per component), so these tests cover the shared mechanics: the
snapshot rules, the enforced missing-policy, and the lookup guards.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from astrai.trainer.optional_extras import (
    COMPONENT_EXTRAS,
    component_extra,
    component_extra_keys,
    load_component_extra,
    require_component_extras,
    snapshot_component_extras,
)


class _Root(nn.Module):
    """Stand-in for a component (critic / reference) with a real state dict."""

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((4,), value))


class _Strategy:
    """Strategy stand-in exposing exactly the table's attributes."""

    def __init__(self, **components):
        self.__dict__.update(components)


def test_table_keys_are_unique_and_declared():
    keys = [entry.key for entry in COMPONENT_EXTRAS]
    assert len(keys) == len(set(keys))
    # The PPO critic is restored as one unit: both its keys are table entries.
    assert set(component_extra_keys("critic", "critic_optimizer")) == {
        "value_model",
        "value_optimizer",
    }
    for key in keys:
        assert component_extra(key).key == key
    with pytest.raises(ValueError, match="unknown component extra"):
        component_extra("no_such_extra")


def test_snapshot_walks_the_table_and_copies_modules_to_cpu():
    """A strategy owning all three components snapshots all three keys, with
    module state stored as detached CPU tensors."""
    strategy = _Strategy(
        critic=_Root(1.0),
        critic_optimizer=torch.optim.SGD(_Root(1.0).parameters(), lr=0.1),
        ref_model=_Root(2.0),
    )
    extra = snapshot_component_extras(strategy, SimpleNamespace())

    assert set(extra) == {"value_model", "value_optimizer", "reference_model"}
    for module_key in ("value_model", "reference_model"):
        for tensor in extra[module_key].values():
            assert tensor.device.type == "cpu"
            assert not tensor.requires_grad
    # Optimizer state stays verbatim: that is what load_state_dict expects.
    assert "param_groups" in extra["value_optimizer"]


def test_snapshot_skips_absent_components_and_gated_entries():
    """Components the strategy does not own are skipped; the reference is
    gated by its config flag (default on)."""
    critic_only = _Strategy(critic=_Root(), critic_optimizer=None)
    assert set(snapshot_component_extras(critic_only, None)) == {"value_model"}

    full = _Strategy(
        critic=_Root(),
        critic_optimizer=torch.optim.SGD(_Root().parameters(), lr=0.1),
        ref_model=_Root(),
    )
    gated = snapshot_component_extras(full, SimpleNamespace(save_reference_model=False))
    assert "reference_model" not in gated
    assert set(gated) == {"value_model", "value_optimizer"}
    # A missing config keeps the default (save the reference).
    assert "reference_model" in snapshot_component_extras(full, None)


def test_load_restores_the_declared_key_into_the_target():
    source = _Root(3.0)
    extra = {"reference_model": source.state_dict()}
    target = _Root(0.0)

    assert load_component_extra(extra, "reference_model", target) is True

    torch.testing.assert_close(target.weight, source.weight)


def test_missing_extra_is_fatal_unless_the_escape_flag_is_set():
    """`reference_model` is the declared case: fatal by default, downgraded to
    a warning by its allow-missing flag, and the message names the key plus
    both flags."""
    target = _Root(0.0)

    with pytest.raises(ValueError, match="reference_model") as excinfo:
        load_component_extra({}, "reference_model", target, SimpleNamespace())
    message = str(excinfo.value)
    assert "save_reference_model" in message
    assert "allow_reference_reanchor" in message

    allowed = SimpleNamespace(allow_reference_reanchor=True)
    assert load_component_extra({}, "reference_model", target, allowed) is False
    torch.testing.assert_close(target.weight, torch.zeros(4))


def test_require_component_extras_lists_every_missing_key():
    """Fail-fast validation keeps the message the PPO resume path relies on."""
    present = {"value_model": {}}

    with pytest.raises(ValueError, match="missing extras: value_optimizer"):
        require_component_extras(
            present,
            component_extra_keys("critic", "critic_optimizer"),
            strategy="online_ppo",
            requirement="critic state",
        )

    require_component_extras(
        {"value_model": {}, "value_optimizer": {}},
        component_extra_keys("critic", "critic_optimizer"),
        strategy="online_ppo",
        requirement="critic state",
    )
