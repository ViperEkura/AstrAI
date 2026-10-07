"""CPU batch slicing and rollout assembly shared by trainer and workers."""

from typing import Dict, List, Tuple

import torch
from torch import Tensor

from astrai.trainer.rollout.types import RawRollout, RolloutVersionError


def slice_batch(batch: Dict, indices: List[int], total: int) -> Dict:
    result = {}
    for key, value in batch.items():
        if isinstance(value, Tensor) and value.shape[0] == total:
            result[key] = value[indices]
        elif isinstance(value, (list, tuple)) and len(value) == total:
            result[key] = [value[index] for index in indices]
        else:
            result[key] = value
    return result


def merge_rollouts(parts: List[Tuple[List[int], RawRollout]], total: int) -> RawRollout:
    """Restore input order while retaining prompt-left and response-right pads."""
    first = parts[0][1]
    versions = {raw.policy_version for _, raw in parts}
    if versions != {first.policy_version}:
        raise RolloutVersionError(f"mixed rollout versions: {sorted(versions)}")
    group_size = first.responses.shape[1]
    prompt_len = max(raw.prompts.shape[1] for _, raw in parts)
    response_len = max(raw.responses.shape[2] for _, raw in parts)
    prompts = torch.zeros(total, prompt_len, dtype=torch.long)
    prompt_mask = torch.zeros(total, prompt_len, dtype=torch.bool)
    responses = torch.zeros(total, group_size, response_len, dtype=torch.long)
    response_mask = torch.zeros(total, group_size, response_len, dtype=torch.bool)
    logprobs_old = torch.zeros(total, group_size, response_len, dtype=torch.float)
    prompt_texts = [""] * total
    response_texts = [[] for _ in range(total)]
    finish_reasons = [[] for _ in range(total)]
    for indices, raw in parts:
        local_prompt_len = raw.prompts.shape[1]
        local_response_len = raw.responses.shape[2]
        prompts[indices, -local_prompt_len:] = raw.prompts
        prompt_mask[indices, -local_prompt_len:] = raw.prompt_mask
        responses[indices, :, :local_response_len] = raw.responses
        response_mask[indices, :, :local_response_len] = raw.response_mask
        logprobs_old[indices, :, :local_response_len] = raw.logprobs_old
        for local, global_index in enumerate(indices):
            prompt_texts[global_index] = raw.prompt_texts[local]
            response_texts[global_index] = raw.response_texts[local]
            finish_reasons[global_index] = raw.finish_reasons[local]
    return RawRollout(
        prompts=prompts,
        prompt_mask=prompt_mask,
        responses=responses,
        response_mask=response_mask,
        logprobs_old=logprobs_old,
        policy_version=first.policy_version,
        prompt_texts=prompt_texts,
        response_texts=response_texts,
        finish_reasons=finish_reasons,
    )
