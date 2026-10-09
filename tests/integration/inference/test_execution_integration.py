import threading
from collections import Counter

import pytest

from astrai.inference.core.events import (
    FINISH_LENGTH,
    FINISH_STOP_TOKEN,
    RequestError,
    RequestFinished,
    TokenDelta,
)
from tests.support.inference import make_cpu_scheduler


@pytest.mark.parametrize("page_size", [None, 2, 4])
@pytest.mark.parametrize("overlap", [False, True])
def test_online_batch_changes_preserve_events_and_greedy_tokens(
    test_model, test_tokenizer, page_size, overlap
):
    model = test_model["model"].eval()
    cache_options = (
        {} if page_size is None else {"page_size": page_size, "kv_tokens": 64}
    )
    scheduler = make_cpu_scheduler(
        model,
        test_tokenizer,
        max_batch_size=2,
        max_seq_len=24,
        enable_overlap=overlap,
        token_budget=3,
        **cache_options,
    )
    events = []
    terminal_ids = set()
    finished = threading.Event()
    prompts = {"short": [7, 8, 9, 10, 11], "long": [7, 8, 9, 10, 12, 13, 14]}
    limits = {"short": 2, "long": 6}

    def sink(batch):
        events.extend(batch)
        terminal_ids.update(
            event.request_id
            for event in batch
            if isinstance(event, (RequestFinished, RequestError))
        )
        if terminal_ids == set(prompts):
            finished.set()

    scheduler.set_event_sink(sink)
    for request_id, prompt in prompts.items():
        scheduler.add_request(
            "",
            prompt_ids=prompt,
            request_id=request_id,
            max_tokens=limits[request_id],
            temperature=0,
        )
    try:
        scheduler.start()
        assert finished.wait(20), f"generation did not terminate: {events!r}"
    finally:
        scheduler.stop()

    assert not any(isinstance(event, RequestError) for event in events)
    terminals = [event for event in events if isinstance(event, RequestFinished)]
    assert Counter(event.request_id for event in terminals) == {
        request_id: 1 for request_id in prompts
    }
    for request_id, prompt in prompts.items():
        request_events = [event for event in events if event.request_id == request_id]
        assert isinstance(request_events[-1], RequestFinished)
        deltas = [event for event in request_events if isinstance(event, TokenDelta)]
        assert [event.sequence_no for event in deltas] == list(
            range(1, len(deltas) + 1)
        )
        terminal = request_events[-1]
        # The fixture model is randomly initialized: greedy decoding may
        # emit a stop token (<pad>) before max_tokens. That is a correct
        # terminal, not a failure — both reasons must keep the invariants.
        assert terminal.finish_reason in (FINISH_LENGTH, FINISH_STOP_TOKEN)
        assert terminal.completion_tokens == len(deltas)
        if terminal.finish_reason == FINISH_LENGTH:
            assert len(deltas) == limits[request_id]
        else:
            # A stop token sampled on the FINAL allowed token is a normal
            # full-length generation that happened to end on a stop id.
            assert len(deltas) <= limits[request_id]
            assert deltas[-1].token_id in scheduler.stop_ids
        assert terminal.prompt_tokens == len(prompt)
        expected = scheduler.run_batch(
            [prompt],
            max_tokens=limits[request_id],
            temperature=0,
            return_details=True,
        )[0]
        # The online run stops at the first sampled stop token; run_batch
        # keeps decoding to max_tokens, so compare only the emitted prefix.
        assert [event.token_id for event in deltas] == expected.token_ids[: len(deltas)]
