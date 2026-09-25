"""Slot addressing for the fp8 per-module state.

The C++ op (``csrc/kernels/gemm/fp8_linear.cu``) keeps per-weight state —
three delayed-scaling rings, the version-keyed weight-cast cache — in a
process-wide registry. Its *original* key is the weight's
``(data_ptr, shape, dtype)``, discovered lazily on first use, and its
checkpoint snapshot binds entries to modules by ``(shape, dtype)`` in
registration order. Both are address-based: two same-shaped linears are told
apart only by the order they happened to run, and a replaced weight (TP
sharding, FSDP, a graph-capture buffer) is a different address, so its history
is silently thrown away.

This module gives every fp8-capable module a *name* instead. It walks
``named_modules()`` once and assigns each ``Linear`` a ``Fp8Slot``: a stable
module path (``layers.3.attention.q_proj``) that survives weight replacement,
plus the parsed role (``module_type``/``tensor_type``) that per-role policy
will need. The path — not the compact ``slot_id`` — is the identity the
snapshot binds on, so a model whose construction order changes cannot
cross-wire state between layers.

Attaching is done from the outside, the same way TP shards modules
(``astrai/parallel/tp.py``): modules get a ``_fp8_slot`` attribute, and
``astrai/model/**`` is not touched. The mapping from a weight tensor to its
slot lands in ``by_weight``, keyed by ``id(weight)`` with a strong reference
held alongside — the identity trick the C++ ``ActivationCast`` anchor uses,
for the same reason: a recycled address must not fake a hit.

Nothing here allocates device state or talks to the extension: it is a pure
name table, so it is safe to import (and to test) without CUDA.
"""

import weakref
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import torch

from astrai.extension.loader import get_module, is_available

__all__ = [
    "Fp8Slot",
    "SlotTable",
    "assign_slots",
    "clear_slots",
    "fp8_slot_of",
    "refresh_slots",
    "slot_table",
    "sync_slots",
]

# Module path prefixes that wrappers add. ``torch.compile`` prefixes
# ``_orig_mod.`` (see ``astrai/parallel/executor.py``); DDP wraps in
# ``module.``. Stripping them keeps one model's slots identical whether or not
# it is compiled or wrapped.
_PATH_PREFIXES = ("_orig_mod.", "module.")


def _canonical_path(name: str) -> str:
    """Drop wrapper prefixes so a path is the same before and after wrapping."""
    changed = True
    while changed:
        changed = False
        for prefix in _PATH_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
                break
    return name


_linear_types_cache: Optional[tuple] = None


def _linear_types() -> tuple:
    """Model classes whose forward is an ``aten::linear`` call.

    Imported lazily: the extension package must stay importable on a box with
    no model package (and importing it must not pull the model stack).
    ``LoRALinear`` wraps a base ``Linear`` and re-registers the *same* weight
    parameter, so a slot assigned before injection stays valid after it.
    """
    global _linear_types_cache
    if _linear_types_cache is None:
        from astrai.model.components.linear import Linear

        types: List[type] = [Linear]
        try:
            from astrai.model.components.lora import LoRALinear

            types.append(LoRALinear)
        except ImportError:  # pragma: no cover - model package always present
            pass
        _linear_types_cache = tuple(types)
    return _linear_types_cache


def _linear_weight(module: torch.nn.Module) -> Optional[torch.Tensor]:
    """The 2-D weight of a module that dispatches through ``aten::linear``."""
    if not isinstance(module, _linear_types()):
        return None
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or weight.dim() != 2:
        return None
    return weight


@dataclass(frozen=True)
class Fp8Slot:
    """One quantized slot: a module path plus its parsed role.

    ``name`` is the module path (e.g. ``layers.3.attention.q_proj``) and is the
    *stable identity* — the snapshot binds on it. ``slot_id`` is a compact
    index handed to C++ for per-call lookup; it follows ``named_modules()``
    order and carries no meaning of its own.

    ``module_type``/``tensor_type`` mirror TE's ``QuantizerRole``: the kind of
    parent the linear hangs under (``attention``/``mlp``/``""``) and the child
    name (``q_proj``/``up``/``lm_head``). ``pattern`` is the TE-plan-style
    glob (``layers.*.attention.q_proj``) a per-role policy would match on.
    """

    slot_id: int
    name: str
    module_type: str
    tensor_type: str

    @property
    def pattern(self) -> str:
        """Path with numeric indices globbed (``layers.*.mlp.up``)."""
        return ".".join(
            "*" if segment.isdigit() else segment for segment in self.name.split(".")
        )


def _parse_role(name: str) -> tuple:
    """``layers.3.attention.q_proj`` -> ``("attention", "q_proj")``.

    The module type is the last non-numeric segment of the parent path, so MoE
    experts (``...routed_experts.2.up``) report ``routed_experts`` rather than
    the expert index. A top-level projection (``lm_head``) has no parent.
    """
    parent, _, child = name.rpartition(".")
    module_type = ""
    for segment in reversed(parent.split(".")):
        if segment and not segment.isdigit():
            module_type = segment
            break
    return module_type, (child or name)


class SlotTable:
    """Weight-identity -> slot map for one model.

    ``by_weight`` is keyed by ``id(weight)`` rather than by the tensor itself:
    ``Tensor.__eq__`` is elementwise, so a dict keyed on tensors would compare
    values on a hash collision. Identity hashing needs the address to stay
    unique, which is what ``_keepalive`` is for — it holds a strong reference
    to every registered weight, so no address can be recycled into a false hit.
    ``rebind()`` is the release valve: it drops references to weights the model
    no longer owns.
    """

    def __init__(self) -> None:
        self._slots: List[Fp8Slot] = []
        self._by_name: Dict[str, Fp8Slot] = {}
        self._by_slot: Dict[int, Fp8Slot] = {}
        self._by_weight: Dict[int, Fp8Slot] = {}
        self._keepalive: List[torch.Tensor] = []
        # slot id -> (module, weight attribute), weakly held so a dead model
        # does not keep itself alive through the table.
        self._owners: Dict[int, tuple] = {}
        self._model_ref = None

    def __len__(self) -> int:
        return len(self._slots)

    def __iter__(self) -> Iterator[Fp8Slot]:
        return iter(self._slots)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    # -- construction -------------------------------------------------------

    def assign(self, model: torch.nn.Module) -> "SlotTable":
        """Walk ``model`` and (re)build the table.

        Re-assigning an unchanged model reuses the existing ``slot_id`` for
        each path, so ids stay put across a rebuild; a genuinely new path gets
        the next free id. One weight shared by two modules (tied LM head) is
        one slot, named after the first module that reaches it.
        """
        slots: List[Fp8Slot] = []
        by_name: Dict[str, Fp8Slot] = {}
        by_weight: Dict[int, Fp8Slot] = {}
        keepalive: List[torch.Tensor] = []
        owners: Dict[int, tuple] = {}
        # Ids are handed out monotonically and never reused, so a path that
        # disappears and a new one that appears cannot collide on an id. The
        # floor is the previous table's maximum: a pass only sees the slots it
        # has reached so far, which lags behind ids assigned by an earlier pass.
        next_id = max((slot.slot_id for slot in self._slots), default=-1) + 1

        for raw_name, module in model.named_modules():
            weight = _linear_weight(module)
            if weight is None:
                continue
            name = _canonical_path(raw_name)
            slot = self._by_name.get(name)
            if slot is None:
                module_type, tensor_type = _parse_role(name)
                slot = Fp8Slot(next_id, name, module_type, tensor_type)
            next_id = max(next_id, slot.slot_id + 1)
            slots.append(slot)
            by_name[name] = slot
            # Tied weights: one state, so the second module must not overwrite
            # the first binding with a different slot.
            if id(weight) not in by_weight:
                by_weight[id(weight)] = slot
                keepalive.append(weight)
            owners[slot.slot_id] = (weakref.ref(module), "weight")
            module._fp8_slot = slot.slot_id  # external attach (TP-style)

        self._slots = slots
        self._by_name = by_name
        self._by_slot = {slot.slot_id: slot for slot in slots}
        self._by_weight = by_weight
        self._keepalive = keepalive
        self._owners = owners
        self._model_ref = weakref.ref(model)
        return self

    def rebind(self, model: torch.nn.Module) -> None:
        """Re-point weight identities after parameters were replaced.

        TP sharding, FSDP and graph-capture buffers swap the ``Parameter``
        object, which changes ``id(weight)`` while the module path stays put.
        Re-assigning restores the mapping and releases the stale references.
        """
        self.assign(model)

    def stale(self) -> int:
        """How many slots no longer hold the weight tensor they registered.

        A stale table is not a correctness problem — an unmatched weight falls
        back to the address-keyed path — but the module loses its history, so
        :func:`refresh_slots` repairs it at the next fp8 region entry. Reads one
        attribute per slot: ~10us on a 1B model, once per region.
        """
        count = 0
        for slot_id, (module_ref, attr) in self._owners.items():
            module = module_ref()
            if module is None:
                continue
            if self._by_weight.get(
                id(getattr(module, attr, None))
            ) is not self._by_slot.get(slot_id):
                count += 1
        return count

    def refresh(self) -> bool:
        """Re-assign from the model when a weight was swapped under it.

        True when something had changed. A no-op when the model is gone (the
        table outlives a throwaway model) or nothing moved, which keeps the
        common path free.
        """
        model = self._model_ref() if self._model_ref is not None else None
        if model is None or not self.stale():
            return False
        self.assign(model)
        return True

    def clear(self) -> None:
        """Drop every slot and reference (tests, teardown)."""
        self._slots = []
        self._by_name = {}
        self._by_slot = {}
        self._by_weight = {}
        self._keepalive = []
        self._owners = {}
        self._model_ref = None

    # -- lookup -------------------------------------------------------------

    def by_name(self, name: str) -> Optional[Fp8Slot]:
        return self._by_name.get(_canonical_path(name))

    def by_weight(self, weight: torch.Tensor) -> Optional[Fp8Slot]:
        return self._by_weight.get(id(weight))

    def slot_of(self, weight: torch.Tensor) -> int:
        """Compact id for ``weight``, or ``-1`` when it was never assigned.

        ``-1`` is the C++ side's "no slot" sentinel: an unassigned weight (a
        bare call, a bench, a model that never went through :func:`assign_slots`)
        keeps the original address-keyed behavior.
        """
        slot = self._by_weight.get(id(weight))
        return -1 if slot is None else slot.slot_id

    def name_of(self, weight: torch.Tensor) -> str:
        slot = self._by_weight.get(id(weight))
        return "" if slot is None else slot.name

    def describe(self) -> List[str]:
        """One ``slot_id name`` line per slot (debug/tests)."""
        return [f"{slot.slot_id} {slot.name}" for slot in self._slots]


# The process-wide table the dispatcher consults. One model per process is the
# training assumption; re-assigning a different model replaces the contents.
_slots = SlotTable()


def sync_slots(table: Optional[SlotTable] = None) -> None:
    """Publish ``table`` (the active one by default) to the C++ state.

    C++ needs slot id -> module path to name its snapshot entries; keeping that
    map there means only an ``int64`` crosses the per-call boundary. A no-op on
    a box without the extension — the table itself is a pure name map, so slot
    assignment never requires CUDA.
    """
    if not is_available("gemm"):
        return
    target = _slots if table is None else table
    get_module("gemm").fp8_set_slots(
        [(slot.slot_id, slot.name, slot.pattern) for slot in target]
    )


def slot_table() -> SlotTable:
    """The active table (queried by the ``aten::linear`` override)."""
    return _slots


def assign_slots(model: torch.nn.Module) -> SlotTable:
    """Assign slots for ``model``, install them as the active table, publish."""
    table = _slots.assign(model)
    sync_slots(table)
    return table


def clear_slots() -> None:
    """Forget every slot (tests / reconfiguration)."""
    _slots.clear()
    sync_slots()


def refresh_slots() -> bool:
    """Repair a stale table after a parameter swap; True when it changed.

    Called on fp8 region entry: the C++ rebind path only fires when the same
    slot id arrives with a different weight tensor, which is exactly what a
    stale Python mapping cannot produce.
    """
    if _slots.refresh():
        sync_slots(_slots)
        return True
    return False


def fp8_slot_of(weight: torch.Tensor) -> int:
    """Slot id of a linear weight, ``-1`` when unassigned.

    The seam a future producer (norm/activation epilogue writing fp8 directly)
    uses to find the consuming linear's state without a tensor-identity
    registry of its own.
    """
    return _slots.slot_of(weight)
