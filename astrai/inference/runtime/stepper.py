"""One-token advancement primitive shared by every scheduling mode."""

from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

from astrai.extension import AttentionBackend, attn_backend
from astrai.inference.cache import PagePool, TaskCacheManager
from astrai.inference.metrics import MetricsCollector
from astrai.inference.runtime.executor import Executor
from astrai.inference.task import Task, TaskStatus


class Stepper:
    """Advance every active task by one token (prefill + decode).

    Single shared primitive for both the continuous-batching loop and the
    synchronous ``run_batch`` path, so the two cannot drift.

    Tasks must already be allocated in the KV cache. Tasks without output
    are prefilled first and sample their first token from the final prompt
    position. Tasks with output extend the cache by one position and decode
    from their latest generated token.
    """

    def __init__(
        self,
        pool: PagePool,
        task_cache: TaskCacheManager,
        executor: Executor,
        metrics: MetricsCollector,
    ):
        self._pool = pool
        self._task_cache = task_cache
        self._executor = executor
        self._metrics = metrics

    @staticmethod
    def _task_backend_groups(tasks: List[Task]):
        groups = {}
        for task in tasks:
            groups.setdefault(task.backend, (task.backend, []))[1].append(task)
        return groups.values()

    def step(
        self, tasks: List[Task], return_logprobs: bool = False
    ) -> Tuple[List[Task], List[Task]]:
        """Advance ``tasks`` by one token.

        Args:
            tasks: Active tasks to advance by one token.
            return_logprobs: Forwarded to the executor; per-token logprobs
                are recorded on each task's ``output_logprobs``.

        Returns:
            ``(decoded, aborted)``: tasks that produced a new token (its ID
            already appended to ``output_ids``) and tasks that hit the
            sequence cap and were marked ``ABORTED``.
        """
        to_prefill = [t for t in tasks if not t.prefill_done and t.prompt_ids]
        prefilled_ids = set()
        produced: List[Task] = []
        if to_prefill:
            for t in to_prefill:
                t.input_tokens = len(t.prompt_ids)

            groups: Dict[Tuple[int, Optional[AttentionBackend]], List[Task]] = {}
            for t in to_prefill:
                start_pos = min(
                    self._task_cache.task_cached(t.task_id), len(t.prompt_ids) - 1
                )
                groups.setdefault((start_pos, t.backend), []).append(t)

            for (start_pos, _), group in groups.items():
                backend = group[0].backend
                backend_context = (
                    attn_backend(backend) if backend is not None else nullcontext()
                )
                with (
                    backend_context,
                    self._metrics.record([t.task_id for t in group], "prefill"),
                ):
                    prefilled, step_out = self._executor.execute_prefill(
                        group, start_pos=start_pos, return_logprobs=return_logprobs
                    )

                for t, out in zip(prefilled, step_out):
                    t.output_ids.append(out[0] if return_logprobs else out)
                    t.output_tokens += 1
                    t.mark_prefill_done()
                    prefilled_ids.add(t.task_id)
                    produced.append(t)

                start_logical_page = start_pos // self._pool.page_size
                for t in group:
                    self._task_cache.task_record_hashes(
                        t.task_id, t.prompt_ids, start_logical_page
                    )

        decoded: List[Task] = []
        aborted: List[Task] = []
        for t in tasks:
            if t.task_id in prefilled_ids:
                continue
            if self._task_cache.task_extend(t.task_id, t.next_pos):
                decoded.append(t)
            else:
                t.status = TaskStatus.ABORTED
                aborted.append(t)

        for backend, group in self._task_backend_groups(decoded):
            backend_context = (
                attn_backend(backend) if backend is not None else nullcontext()
            )
            with (
                backend_context,
                self._metrics.record([t.task_id for t in group], "decode"),
            ):
                step_out = self._executor.execute_decode(
                    group, return_logprobs=return_logprobs
                )
            for t, out in zip(group, step_out):
                t.output_ids.append(out[0] if return_logprobs else out)
                t.output_tokens += 1
                t.advance_kv()
                produced.append(t)

        return produced, aborted
