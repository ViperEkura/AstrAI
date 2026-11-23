"""Plan prefill windows and decode slots without executing a model."""

from typing import List, Optional, Tuple

from astrai.inference.contracts import ExecutionRequest
from astrai.inference.core.cache import KVCacheManager
from astrai.inference.core.request import Request


class StepPlanner:
    """Share the step budget across prefill windows and reserve decode KV."""

    def __init__(self, kv_manager: KVCacheManager, token_budget: Optional[int] = None):
        if token_budget is not None and token_budget <= 0:
            raise ValueError("token_budget must be positive or None")
        self.kv_manager = kv_manager
        self._token_budget = token_budget

    def plan(
        self, requests: List[Request]
    ) -> Tuple[List[ExecutionRequest], List[Request]]:
        entries = []
        groups = {}
        decode = []
        for request in requests:
            if not request.prefill_complete:
                start = min(
                    max(
                        request.num_computed_tokens,
                        request.cached_tokens,
                    ),
                    len(request.prompt_ids) - 1,
                )
                groups.setdefault(start, []).append(request)
            else:
                decode.append(request)

        remaining = self._token_budget
        for start, group in groups.items():
            for request in group:
                need = len(request.prompt_ids) - start
                count = need if remaining is None else min(need, remaining)
                if count <= 0:
                    break
                entries.append(request.execution("prefill", start, count))
                if remaining is not None:
                    remaining -= count

        failed = []
        if decode:
            extended = self.kv_manager.alloc(decode, [r.next_pos + 1 for r in decode])
            for request, ok in zip(decode, extended):
                if ok:
                    entries.append(request.execution("decode", request.next_pos, 1))
                else:
                    failed.append(request)
        return entries, failed
