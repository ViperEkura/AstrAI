import logging
from typing import List, Optional

import torch

from astrai.inference.core.cache import KVCache
from astrai.inference.core.task import Task
from astrai.inference.sample import sample
from astrai.model.automodel import AutoModel
from astrai.tokenize.tokenizer import AutoTokenizer

logger = logging.getLogger(__name__)


class Executor:
    """Model forward passes for prefill and decode phases."""

    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        kv_cache: KVCache,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache = kv_cache
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

    def execute_prefill(self, tasks: List[Task], prompt_len: int, start_pos: int = 0):
        if start_pos >= prompt_len:
            return

        tasks = sorted(tasks, key=lambda t: t.task_id)
        batch_sz = len(tasks)

        input_ids = torch.tensor(
            [t.prompt_ids[start_pos:prompt_len] for t in tasks],
            dtype=torch.long,
            device=self.device,
        )

        task_ids = [t.task_id for t in tasks]

        with torch.inference_mode():
            self.model(
                input_ids,
                position_ids=torch.arange(
                    start_pos, prompt_len, dtype=torch.long, device=self.device
                )
                .unsqueeze(0)
                .expand(batch_sz, -1),
                paged_cache=self.kv_cache.bind_tasks(task_ids, prompt_len, self.device),
            )

    def execute_decode(self, tasks: List[Task]) -> List[int]:
        if not tasks:
            return []

        input_ids = torch.tensor(
            [t.output_ids[-1] if t.output_ids else t.prompt_ids[-1] for t in tasks],
            dtype=torch.long,
            device=self.device,
        )

        position_ids = torch.tensor(
            [t.next_pos for t in tasks], dtype=torch.long, device=self.device
        )
        total_len = position_ids.max().item() + 1

        task_ids = [t.task_id for t in tasks]

        temperatures = torch.tensor([t.temperature for t in tasks], device=self.device)
        top_ks = torch.tensor([t.top_k for t in tasks], device=self.device)
        top_ps = torch.tensor([t.top_p for t in tasks], device=self.device)
        freq_penalties = torch.tensor(
            [t.frequency_penalty for t in tasks], device=self.device
        )

        history_lists = []
        mask_lists = []
        for t in tasks:
            window = t.rep_window
            prompt_part = t.prompt_ids[-window:]
            ids = prompt_part + t.output_ids
            history_lists.append(ids)
            mask_lists.append([True] * len(ids))

        max_len = max(len(h) for h in history_lists)
        padded_ids = torch.zeros(
            len(tasks), max_len, dtype=torch.long, device=self.device
        )
        padded_mask = torch.zeros(
            len(tasks), max_len, dtype=torch.bool, device=self.device
        )
        for i, (h, m) in enumerate(zip(history_lists, mask_lists)):
            padded_ids[i, : len(h)] = torch.tensor(
                h, dtype=torch.long, device=self.device
            )
            padded_mask[i, : len(m)] = torch.tensor(
                m, dtype=torch.bool, device=self.device
            )

        with torch.inference_mode():
            outputs = self.model(
                input_ids.unsqueeze(1),
                paged_cache=self.kv_cache.bind_tasks(
                    task_ids,
                    total_len,
                    self.device,
                    write_positions=position_ids,
                ),
                position_ids=position_ids.unsqueeze(1),
            )
            logits = outputs["logits"][:, -1, :]

        return sample(
            logits,
            temperature=temperatures,
            top_k=top_ks,
            top_p=top_ps,
            frequency_penalty=freq_penalties,
            input_ids=padded_ids,
            input_mask=padded_mask,
        ).tolist()
