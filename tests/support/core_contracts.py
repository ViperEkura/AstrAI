"""Deterministic CPU gates for execution identity, lifecycle and resource ownership."""

import pytest

from tests.support.inference import make_cpu_model, make_cpu_scheduler
from tests.support.tokenizers import FakeTokenizer


@pytest.fixture
def scheduler():
    model = make_cpu_model()
    model.generation_config = {"eos_token_id": 0}
    tokenizer = FakeTokenizer()
    tokenizer.eos_token_id = 0
    sched = make_cpu_scheduler(model, tokenizer, max_batch_size=4, device="cpu")
    events = []
    sched.set_event_sink(events.extend)
    yield sched, events
    sched.stop()
