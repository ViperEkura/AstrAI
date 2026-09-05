"""Semantic contracts and diagnostics for mixture-of-experts routing."""

from astrai.moe.route_metrics import (
    RouteAlignmentMetricsV0,
    compare_route_traces,
    compute_topk_margin,
)
from astrai.moe.route_trace import (
    ROUTE_TRACE_SCHEMA_VERSION,
    PaddingLayout,
    RouteIdentityV0,
    RouterSchemaV0,
    RouteTokenLayoutV0,
    RouteTraceCodecV0,
    RouteTraceLevel,
    RouteTraceV0,
    RouteTraceValidationError,
    SelectedWeightSemantics,
    TokenSpanKind,
    canonical_json_digest,
    pack_route_ids,
    recommended_route_id_dtype,
    require_compatible_route_trace,
    require_semantically_aligned_traces,
    validate_route_trace,
)

__all__ = [
    "ROUTE_TRACE_SCHEMA_VERSION",
    "PaddingLayout",
    "RouteAlignmentMetricsV0",
    "RouteIdentityV0",
    "RouteTokenLayoutV0",
    "RouteTraceCodecV0",
    "RouteTraceLevel",
    "RouteTraceV0",
    "RouteTraceValidationError",
    "RouterSchemaV0",
    "SelectedWeightSemantics",
    "TokenSpanKind",
    "canonical_json_digest",
    "compare_route_traces",
    "compute_topk_margin",
    "pack_route_ids",
    "recommended_route_id_dtype",
    "require_compatible_route_trace",
    "require_semantically_aligned_traces",
    "validate_route_trace",
]
