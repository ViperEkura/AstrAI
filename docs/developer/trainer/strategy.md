# Objectives and registration

The strategy package owns algorithm-specific batch preparation, losses and
objective state. Trainer calls the common BaseStrategy interface and owns the
optimizer loop. Shared tensor operations live in ops.py.

## Source modules

| Module | Responsibility |
| --- | --- |
| base.py | Loss outputs, rollout conversion and algorithm post-update hooks |
| factory.py | Registry, capability declarations and strict internal construction |
| ops.py | ForwardResult and LossOutput, log probabilities, values, GAE and common rollout tensor operations |
| supervised.py | Shared cross-entropy path, sequence and SFT masking/reduction |
| dpo.py | Chosen/rejected preference objective and frozen reference |
| grpo.py | Group advantages, clipped objective and old-policy state |
| ppo.py | Actor/critic objective, advantages and value targets |

## Loss boundary

Trainer obtains scored results from `context.rollout`, calls
`prepare_from_rollout(result)`, and schedules replay with
`learner_update_slices()`. Only Trainer reads `rl_update_epochs` and
`rl_minibatch_prompts`. PPO advantages and returns stay pinned to their rollout,
so replay neither regenerates responses nor recomputes those targets.

`loss_outputs(batch, microbatch_prompts=None)` yields losses for **one optimizer
update**, with one output by default. Global-window outputs must share a
`normalizer`; Trainer sums their losses before applying the global target count.
Async GRPO splits complete prompt groups and weights each loss by its share of
valid tokens or nonempty responses. It skips empty microbatches and rejects an
entire round without targets.

`compute_loss_output()` returns loss and metrics; `compute_loss()` returns the
scalar loss. `__call__()` only computes the supplied batch's loss.
`prepare_batch()` handles device placement and objective inputs. The token-mean
forward/reduce path supports context parallelism: CPStrategy explicitly implements
`loss_outputs()` through its own shard path.

After actor update and version publication, Trainer calls `after_update()`.
PPO updates and clears critic gradients outside the actor version lock; offline
GRPO refreshes its old policy. Strategies hold neither rollout runners nor
publishers and do not schedule replay or step the actor optimizer.

`rollout.batching.slice_batch()` preserves complete prompt groups and aligned
Python metadata. `ops.py` supplies chunked log probabilities for the
memory-sensitive path.

## Registry and capabilities

StrategyFactory stores a StrategyCapabilities declaration for each registered
name in the configuration module. TrainConfig validates the registry name and
uses its capabilities for
online, critic and async-round requirements. TrainContextBuilder uses the same
declaration to create reference, old-policy and critic models; rollout setup
uses the online flag. The existing CLI strategy choices are derived from the
registry when its command module is imported.

| Capability | Builder or validation effect |
| --- | --- |
| online | Requires reward_model_fn and enables online rollout validation/setup. |
| reference_model | Creates a frozen reference from the prepared actor. |
| old_model and initialize_old_model | Supplies an old-policy argument; offline GRPO creates a frozen copy, online GRPO starts with None. |
| critic | Requires critic_model_fn and constructs/restores critic state. |
| async_round | Permits the async-round configuration checks. |
| min_group_size | Validates the configured group size. |

Register custom strategies before constructing TrainConfig. Offline
strategies can use the default capability declaration. An online strategy
must declare the capabilities it needs and implement the online strategy
contract. Constructor options should appear in its signature or in
accepted_options; internal StrategyFactory.create rejects unknown
options. The legacy filtering construction API has been removed.

```python
from astrai.trainer.strategy import BaseStrategy, StrategyFactory

@StrategyFactory.register("custom")
class CustomStrategy(BaseStrategy):
    def __init__(self, model, device, scale=1.0, **kwargs):
        super().__init__(model, device, **kwargs)
        self.scale = scale
```

## Extension point

Register a custom strategy before constructing TrainConfig, since its
validator checks the registry. Declare online, reference, old-policy, critic
and async-round requirements in StrategyCapabilities. Internal creation uses
`create()` to reject unknown options. Unknown arguments are no longer
filtered; pass only supported options.
