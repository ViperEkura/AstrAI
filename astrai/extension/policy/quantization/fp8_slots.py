"""Stable module-path identities for FP8 scaling and checkpoint state."""

import weakref
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from astrai.extension.runtime.loader import get_module, is_available
from astrai.model.components.linear import Linear
from astrai.model.components.lora import LoRALinear

# Normalize torch.compile and DDP module paths.
_PATH_PREFIXES = ("_orig_mod.", "module.")


def _canonical_path(name: str) -> str:
    """Strip compile/DDP wrapper prefixes from module paths."""
    changed = True
    while changed:
        changed = False
        for prefix in _PATH_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
                break
    return name


def _linear_weight(module: torch.nn.Module) -> Optional[torch.Tensor]:
    """The 2-D weight of a module that dispatches through ``aten::linear``."""
    if not isinstance(module, (Linear, LoRALinear)):
        return None
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or weight.dim() != 2:
        return None
    return weight


@dataclass(frozen=True)
class Fp8Slot:
    """Module path for checkpoint identity, numeric slot ID for C++ lookup."""

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


def _parse_role(name: str) -> Tuple:
    """Return the last nonnumeric parent component and child name."""
    parent, _, child = name.rpartition(".")
    module_type = ""
    for segment in reversed(parent.split(".")):
        if segment and not segment.isdigit():
            module_type = segment
            break
    return module_type, (child or name)


class SlotTable:
    """Map weight identities to module slots.

    _keepalive prevents Python ID reuse; assign() releases replaced weights."""

    def __init__(self) -> None:
        self._slots: List[Fp8Slot] = []
        self._by_name: Dict[str, Fp8Slot] = {}
        self._by_slot: Dict[int, Fp8Slot] = {}
        self._by_weight: Dict[int, Fp8Slot] = {}
        self._keepalive: List[torch.Tensor] = []
        # Weak owners avoid retaining the model through the table.
        self._owners: Dict[int, Tuple] = {}
        self._model_ref = None

    def __len__(self) -> int:
        return len(self._slots)

    def __iter__(self) -> Iterator[Fp8Slot]:
        return iter(self._slots)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def assign(self, model: torch.nn.Module) -> "SlotTable":
        """Reuse IDs for existing paths; shared weights use the first binding."""
        slots: List[Fp8Slot] = []
        by_name: Dict[str, Fp8Slot] = {}
        by_weight: Dict[int, Fp8Slot] = {}
        keepalive: List[torch.Tensor] = []
        owners: Dict[int, Tuple] = {}
        # New paths must not collide with IDs from the previous table.
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
            # Shared weights keep the first binding.
            if id(weight) not in by_weight:
                by_weight[id(weight)] = slot
                keepalive.append(weight)
            owners[slot.slot_id] = (weakref.ref(module), "weight")
            module._fp8_slot = slot.slot_id

        self._slots = slots
        self._by_name = by_name
        self._by_slot = {slot.slot_id: slot for slot in slots}
        self._by_weight = by_weight
        self._keepalive = keepalive
        self._owners = owners
        self._model_ref = weakref.ref(model)
        return self

    def stale(self) -> int:
        """Count replaced weights whose identity mappings need refreshing."""
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
        """Reassign after a weight replacement; report whether the table changed."""
        model = self._model_ref() if self._model_ref is not None else None
        if model is None or not self.stale():
            return False
        self.assign(model)
        return True

    def clear(self) -> None:
        """Drop slots and retained references."""
        self._slots = []
        self._by_name = {}
        self._by_slot = {}
        self._by_weight = {}
        self._keepalive = []
        self._owners = {}
        self._model_ref = None

    def by_name(self, name: str) -> Optional[Fp8Slot]:
        return self._by_name.get(_canonical_path(name))

    def by_weight(self, weight: torch.Tensor) -> Optional[Fp8Slot]:
        return self._by_weight.get(id(weight))

    def slot_of(self, weight: torch.Tensor) -> int:
        """Return the slot ID, or -1 for C++ address-based fallback."""
        slot = self._by_weight.get(id(weight))
        return -1 if slot is None else slot.slot_id

    def name_of(self, weight: torch.Tensor) -> str:
        slot = self._by_weight.get(id(weight))
        return "" if slot is None else slot.name


# Assigning another model replaces the process-wide table.
_slots = SlotTable()


def sync_slots(table: Optional[SlotTable] = None) -> None:
    """Publish slot names and patterns to C++; no-op without the extension."""
    if not is_available("gemm"):
        return
    target = _slots if table is None else table
    get_module("gemm").fp8_set_slots(
        [(slot.slot_id, slot.name, slot.pattern) for slot in target]
    )


def assign_slots(model: torch.nn.Module) -> SlotTable:
    """Assign and publish the process-wide slot table."""
    table = _slots.assign(model)
    sync_slots(table)
    return table


def clear_slots() -> None:
    """Clear the table and its C++ bindings."""
    _slots.clear()
    sync_slots()


def refresh_slots() -> bool:
    """Repair replaced-weight mappings and publish changes on FP8 region entry."""
    if _slots.refresh():
        sync_slots(_slots)
        return True
    return False


def fp8_slot_of(weight: torch.Tensor) -> int:
    """Return the weight slot ID, or -1 when unassigned."""
    return _slots.slot_of(weight)
