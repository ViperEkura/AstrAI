"""Public extension operators, kernels, and runtime selection."""

from astrai.extension.backend import (
    AttentionDispatcher,
    AttentionSpec,
    apply_rotary_emb,
    attention,
    attention_varlen,
    attention_with_kvcache,
    newton_schulz,
    swiglu,
    symm_out,
    syrk_out,
)
from astrai.extension.kernel import (
    cross_entropy,
    linear_cross_entropy,
)
from astrai.extension.kernel.gemm import (
    facts,
    probe,
    set_log,
    set_planner,
    set_staging,
    set_table,
    state,
    tile_vocabulary,
)
from astrai.extension.kernel.quantize import (
    quantize,
    quantize_scaled,
)
from astrai.extension.policy.gemm import autotune as autotune
from astrai.extension.policy.gemm import plan as plan
from astrai.extension.policy.gemm.plan import PLANNER_MODES
from astrai.extension.policy.quantization import (
    autocast as autocast,
)
from astrai.extension.policy.quantization import (
    fp8_slots as fp8_slots,
)
from astrai.extension.policy.quantization.autocast import (
    FP8Recipe,
    fp8_autocast,
    fp8_format_pair,
    fp8_linear_enable,
    fp8_linear_enabled,
    fp8_load_state_dict,
    fp8_reset,
    fp8_state_dict,
)
from astrai.extension.runtime import loader as loader
from astrai.extension.runtime.context import (
    OpSelection,
    op_backend,
    op_context,
    set_op,
    snapshot_op_selection,
    use_op_selection,
)
from astrai.extension.runtime.loader import KERNEL_NAMES, is_available
from astrai.extension.runtime.operation import (
    ExplicitSelectionError,
    OpDispatcher,
    OpSpec,
)

__all__ = [
    "cross_entropy",
    "linear_cross_entropy",
    "AttentionDispatcher",
    "AttentionSpec",
    "attention",
    "attention_varlen",
    "attention_with_kvcache",
    "is_available",
    "KERNEL_NAMES",
    "apply_rotary_emb",
    "newton_schulz",
    "symm_out",
    "syrk_out",
    "swiglu",
    "FP8Recipe",
    "fp8_autocast",
    "fp8_format_pair",
    "fp8_linear_enable",
    "fp8_linear_enabled",
    "fp8_load_state_dict",
    "fp8_reset",
    "fp8_state_dict",
    "quantize",
    "quantize_scaled",
    "OpSpec",
    "OpDispatcher",
    "OpSelection",
    "op_context",
    "snapshot_op_selection",
    "use_op_selection",
    "ExplicitSelectionError",
    "set_op",
    "op_backend",
    "PLANNER_MODES",
    "plan",
    "facts",
    "probe",
    "set_log",
    "set_planner",
    "set_staging",
    "set_table",
    "state",
    "tile_vocabulary",
]
