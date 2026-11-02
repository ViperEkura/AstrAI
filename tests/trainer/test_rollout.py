import pytest

from tests.support.rollout_cases import (
    ConstantRewardModel,
    make_instruction_batch,
    make_replica_pair,
)

"""Unit tests for the online rollout module."""


import torch

from astrai.trainer.backend import P2PCopyPublisher, ReplicaBackend
from astrai.trainer.context import TrainContext
from astrai.trainer.rollout import (
    RolloutGenerator,
    RolloutRunner,
    SamplingParams,
)
from astrai.trainer.trainer import Trainer
from tests.support.capabilities import skip_no_multi_device
from tests.support.tokenizers import FakeTokenizer


@pytest.mark.cuda
@skip_no_multi_device
def test_replica_backend_generation_and_output_device():
    """A replica on cuda:1 generates under its own scheduler and the
    generator lands rollout tensors on the training device."""
    train_model, replica_model = make_replica_pair()
    replica_model.load_state_dict(train_model.state_dict())
    tokenizer = FakeTokenizer(with_chat_template=True)
    backend = ReplicaBackend(
        model=replica_model,
        tokenizer=tokenizer,
        device="cuda:1",
        max_batch_size=8,
        max_seq_len=128,
    )
    generator = RolloutGenerator(
        backend=backend,
        tokenizer=tokenizer,
        params=SamplingParams(group_size=2, max_tokens=4),
        output_device=torch.device("cuda", 0),
    )

    rollout = generator.generate(make_instruction_batch(n=2))

    assert rollout.responses.shape[:2] == (2, 2)
    assert rollout.responses.device == torch.device("cuda", 0)
    assert rollout.logprobs_old.device == torch.device("cuda", 0)
    # The replica never toggles the shared training model: it stays eval.
    assert replica_model.training is False


@pytest.mark.cuda
@skip_no_multi_device
def test_p2p_publisher_copies_weights_and_advances_version():
    train_model, replica_model = make_replica_pair()
    tokenizer = FakeTokenizer(with_chat_template=True)
    backend = ReplicaBackend(
        model=replica_model,
        tokenizer=tokenizer,
        device="cuda:1",
        max_batch_size=8,
        max_seq_len=128,
    )
    publisher = P2PCopyPublisher(backend)

    with torch.no_grad():
        for p in train_model.parameters():
            p.mul_(2.0).add_(1.0)
    publisher.publish(3, train_model)

    assert backend.policy_version == 3
    for (name, src), (name2, dst) in zip(
        train_model.state_dict().items(), backend.model.state_dict().items()
    ):
        assert name == name2
        assert torch.equal(dst.to(src.device), src)
    # A second publish reuses the cached pairs and keeps versions monotone.
    publisher.publish(4, train_model)
    assert backend.policy_version == 4


@pytest.mark.cuda
@skip_no_multi_device
def test_online_optimizer_step_syncs_replica_atomically():
    """Trainer advances the replica's weights and version
    inside one commit — the cross-GPU rollout contract."""
    train_model, replica_model = make_replica_pair()
    tokenizer = FakeTokenizer(with_chat_template=True)
    backend = ReplicaBackend(
        model=replica_model,
        tokenizer=tokenizer,
        device="cuda:1",
        max_batch_size=8,
        max_seq_len=128,
    )
    runner = RolloutRunner(
        generator=RolloutGenerator(
            backend=backend,
            tokenizer=tokenizer,
            params=SamplingParams(group_size=2, max_tokens=4),
            output_device=torch.device("cuda", 0),
        ),
        reward_model=ConstantRewardModel(),
        rollout_interval=2,
        publishers=[P2PCopyPublisher(backend)],
    )
    for p in train_model.parameters():
        p.grad = torch.ones_like(p)
    optimizer = torch.optim.SGD(train_model.parameters(), lr=0.1)
    before = next(train_model.parameters()).detach().clone()
    Trainer._step_policy(
        TrainContext(model=train_model, optimizer=optimizer, rollout=runner)
    )

    assert not torch.equal(next(train_model.parameters()), before)
    assert backend.policy_version == 1
    assert runner.policy_version == 1
    for (name, src), (name2, dst) in zip(
        train_model.state_dict().items(), backend.model.state_dict().items()
    ):
        assert name == name2
        assert torch.equal(dst.to(src.device), src)
