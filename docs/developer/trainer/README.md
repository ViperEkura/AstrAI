# Trainer internals

For configuration and usage, start with [Training](../../guides/training.md)
and [Distributed training](../../guides/distributed.md). This page describes
where the implementation lives and which module owns each responsibility.

## Layout

```text
astrai/trainer/
  trainer.py             training loop and callback lifecycle
  train_context.py       context, checkpoint/model setup, datasets and assembly
  callbacks/
    base.py              callback protocol and registry
    checkpoint.py        checkpoint persistence
    metrics.py           metric logging and validation
    metric_util.py       metric accessors and gradient SNR tracking
    optimization.py      gradient clipping and activation checkpointing
    progress.py          progress bar
  strategy/
    base.py              common objective lifecycle and loss protocol
    factory.py           strategy registry
    ops.py               shared loss and tensor operations
    supervised.py        sequence and SFT objectives
    dpo.py               DPO objective
    grpo.py              GRPO objective
    ppo.py               PPO objective
  rollout/
    types.py             rollout results and sampling contracts
    generator.py         inference-backed generation
    runner.py            reward scoring, replay cache and evaluation
    setup.py             rollout assembly
  backend.py             colocated and replica rollout backends
  schedule.py            learning-rate schedules
  optional_extras.py     checkpointed component state registry
```

## Ownership

- Trainer owns the epoch/batch loop and invokes callback lifecycle hooks.
- TrainContextBuilder assembles model, optimizer, scheduler, datasets,
  strategy, parallel topology and online rollout. Keep construction order here.
- Each strategy owns its algorithm-specific loss and state. Shared objective
  operations belong in strategy/ops.py.
- Rollout modules own generation and reward evaluation. backend.py owns
  where generation runs and how policy weights reach that backend.
- Callbacks own checkpoint writing, progress display, metric logging and
  validation. Metric helpers live alongside them in callbacks/metric_util.py.

## Import surfaces

Use package exports for supported imports:

- astrai.trainer: Trainer, strategy/scheduler factories, callback protocol
  and registry.
- astrai.trainer.callbacks: built-in callback classes.
- astrai.trainer.strategy: built-in objectives and shared strategy API.
- astrai.trainer.rollout: rollout types, generator, runner and evaluator.

There is no astrai.trainer.train_callback compatibility module; import
callbacks from astrai.trainer.callbacks. The implementation files may change
as responsibilities evolve, so prefer these package-level imports unless
working directly on an implementation.
# FSDP module ownership

`FSDPExecutor` applies `fully_shard` to each distinct `DecoderBlock`, then
to the root model. The root owns remaining embeddings, final norms and
output heads; a tied embedding/head is a single root parameter. Reusing the
same block in a loop wraps it once. Parameters shared across distinct blocks
or between a block and the root are rejected before wrapping.

Configure `executor_kwargs={"reshard_after_forward": True,
"root_reshard_after_forward": False}` to reshard blocks after forward while
keeping the root parameters available through backward. These are the
defaults; each can be set explicitly for a capacity/communication experiment.
The policy follows the bottom-up grouping described in the
[PyTorch FSDP2 documentation](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html).

CPU checks validate the call order, unique parameter ownership, dynamic
root class compatibility and reshard arguments. The two-GPU test compares
updates and canonical weight checkpoints for tied/untied heads. It requires
assigned test GPUs. Real all-gather traces, peak memory and official weights
are separate qualification results; CPU checks do not establish GPU support.
This change does not address full-weight loading peaks or reference-model
replication, and distributed FSDP online rollout remains gated.
