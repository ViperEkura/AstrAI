import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from astrai.model.components.linear import Linear
from astrai.serialization import (
    load_json,
    load_safetensors,
    save_json,
    save_safetensors,
)

logger = logging.getLogger(__name__)

TARGET_MODULES_ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}
TARGET_MODULES_FFN = {"up", "gate", "down"}


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    target_modules: tuple = ("q_proj", "v_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.register_parameter("weight", base.weight)
        self.weight.requires_grad_(False)
        self.bias = base.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.r = r
        self.scaling = alpha / r
        device = self.weight.device
        dtype = self.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(r, self.weight.shape[1], device=device, dtype=dtype) / r)
        self.lora_B = nn.Parameter(torch.zeros(self.weight.shape[0], r, device=device, dtype=dtype))
        self._merged = False

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if not self._merged:
            out += (F.linear(x, self.lora_A) @ self.lora_B.T) * self.scaling
        return out

    def merge(self):
        if self._merged:
            return
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self._merged = True
        del self.lora_A
        del self.lora_B


def _collect_lora_info(model: nn.Module) -> dict:
    names = {}
    for n, m in model.named_modules():
        if isinstance(m, Linear):
            _, _, child = n.rpartition(".")
            names.setdefault(child, []).append(n)
    return names


def _get_lora_count(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, LoRALinear))


def inject_lora(
    model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    target_modules: Optional[Set[str]] = None,
) -> LoRAConfig:
    if target_modules is None:
        target_modules = TARGET_MODULES_ATTN

    available = _collect_lora_info(model)
    injected = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, Linear):
            continue
        parent_name, _, child_name = name.rpartition(".")
        if child_name not in target_modules:
            continue
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha))
        injected += 1

    if injected == 0:
        logger.warning(
            "No LoRA layers injected. Available Linear child names: %s. "
            "target_modules: %s. Check model type and target_modules.",
            sorted(available),
            sorted(target_modules),
        )
    else:
        logger.info("LoRA injected: %d layers (r=%d, alpha=%d)", injected, r, alpha)

    return LoRAConfig(r=r, alpha=alpha, target_modules=tuple(target_modules))


def merge_lora(model: nn.Module):
    n = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            n += 1
    if n == 0:
        logger.warning("No LoRA layers to merge.")
    else:
        logger.info("Merged %d LoRA layers", n)


def save_lora(model: nn.Module, save_dir: str, config: LoRAConfig):
    lora_sd = {
        k: v
        for k, v in model.state_dict().items()
        if k.endswith((".lora_A", ".lora_B"))
    }
    if not lora_sd:
        raise RuntimeError(
            "No LoRA parameters found in model. "
            "The model may not have been injected or was already merged."
        )

    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    save_safetensors(lora_sd, path / "adapter_model.safetensors")
    save_json(asdict(config), path / "adapter_config.json")
    logger.info("LoRA adapter saved to %s (%d keys)", save_dir, len(lora_sd))


def load_lora(model: nn.Module, load_dir: str) -> LoRAConfig:
    path = Path(load_dir)
    raw = load_json(path / "adapter_config.json")
    config = LoRAConfig(
        r=raw["r"], alpha=raw["alpha"], target_modules=tuple(raw["target_modules"])
    )

    existing = _get_lora_count(model)
    if existing > 0:
        logger.warning(
            "Model already has %d LoRA layers. Skipping injection, "
            "loading weights onto existing layers only.",
            existing,
        )
    else:
        inject_lora(
            model,
            r=config.r,
            alpha=config.alpha,
            target_modules=set(config.target_modules),
        )

    weights = load_safetensors(path / "adapter_model.safetensors")
    try:
        missing, unexpected = model.load_state_dict(weights, strict=False)
    except RuntimeError as e:
        msg = str(e)
        if "size mismatch" in msg:
            raise RuntimeError(
                f"LoRA weight shapes do not match the model. "
                f"The adapter config (r={config.r}) may not match the injected layers. "
                f"Original error: {msg}"
            ) from e
        raise

    injected = _get_lora_count(model)
    if injected == 0:
        raise RuntimeError(
            "No LoRA layers found after loading. "
            "Inject LoRA before calling load_lora, or check the adapter config."
        )

    if missing:
        lora_missing = [k for k in missing if "lora" in k]
        if lora_missing:
            raise RuntimeError(
                f"LoRA weight keys not found in model: {lora_missing}. "
                f"The adapter config (r={config.r}) may not match the model."
            )
        logger.debug("LoRA load: %d missing base-weight keys (expected)", len(missing))
    if unexpected:
        logger.warning("LoRA load: %d unexpected keys", len(unexpected))

    logger.info("LoRA adapter loaded from %s", load_dir)
    return config
