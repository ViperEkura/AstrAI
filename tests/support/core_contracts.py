"""Deterministic CPU gates for execution identity, lifecycle and resource ownership."""

import pytest
import torch

from astrai.inference.contracts import (
    ModelRunnerOutput,
    RequestOutput,
)
from astrai.inference.worker.pending import PendingExecution
from tests.support.inference import make_cpu_model, make_cpu_scheduler
from tests.support.tokenizers import FakeTokenizer


@pytest.fixture
def scheduler():
    model = make_cpu_model()
    tokenizer = FakeTokenizer()
    tokenizer.stop_ids = [0]
    sched = make_cpu_scheduler(model, tokenizer, max_batch_size=4, device="cpu")
    events = []
    sched.set_event_sink(events.extend)
    yield sched, events
    sched.stop()


def admit(sched, rid="r", max_tokens=8, prompt=(10, 11)):
    sched.add_request(
        "input", request_id=rid, prompt_ids=list(prompt), max_tokens=max_tokens
    )
    sched.admit_requests()
    return sched._states[rid]


def output(plan, tokens=7, *, reverse=False):
    rows = [
        RequestOutput(r.identity, r.materialized_end, tokens if r.samples else None)
        for r in plan.requests
    ]
    return ModelRunnerOutput(
        plan.step_id,
        plan.policy_version,
        tuple(reversed(rows)) if reverse else tuple(rows),
    )


def pending(plan, token=7, fence=None):
    sampled = tuple(r.identity for r in plan.requests if r.samples)
    return PendingExecution(
        plan,
        tokens=torch.tensor([token] * len(sampled)),
        sampled_identities=sampled,
        completion_event=fence,
    )
