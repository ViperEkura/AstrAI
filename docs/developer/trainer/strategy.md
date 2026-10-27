# Objectives and registration

The strategy package owns algorithm-specific batch preparation, losses and
objective state. Trainer calls the common BaseStrategy interface and owns the
optimizer loop. Shared tensor operations live in ops.py.

## Source modules

| Module | Responsibility |
| --- | --- |
| base.py | Offline/online lifecycle, loss output, rollout conversion and optimizer-step hooks |
| factory.py | Registry, capability declarations and strict internal construction |
| ops.py | ForwardResult and LossOutput, log probabilities, values, GAE and common rollout tensor operations |
| supervised.py | Shared cross-entropy path, sequence and SFT masking/reduction |
| dpo.py | Chosen/rejected preference objective and frozen reference |
| grpo.py | Group advantages, clipped objective and old-policy state |
| ppo.py | Actor/critic objective, advantages and value targets |

## Loss boundary

BaseStrategy.prepare_batch() places objective inputs on the target device;
strategy overrides may synthesize missing inputs.
prepare_rollout_batch() collects or reuses a rollout, refreshes objective state
only when the rollout is fresh, and calls prepare_from_rollout() once before
learner replay. compute_loss_output() returns loss and metrics.
The token-mean forward/reduce path supports context parallelism; each
algorithm retains its own reduction semantics. training_steps() and
training_updates() express microbatch or update shape while Trainer retains
commit ownership. ops.py also supplies chunked log-probability calculation
for the memory-sensitive path.

For offline training with the global cursor, one data microbatch must yield one
`training_updates()` group. That group may contain several loss outputs if they
share the same `normalizer`; their losses are added before the global target
count is applied. The trainer rejects additional groups because it owns the
optimizer-window boundary. Synchronous online training collects the rollout
window first and schedules learner updates in the trainer, then calls
`compute_loss_output()` on each prepared chunk.

`step_plan.learner_update_slices()` supplies the common replay-epoch and prompt
minibatch order for synchronous Trainer windows and direct strategy iteration.
Its final-update marker controls when a complete global window can commit its
data cursor. Slicing never divides a prompt's response group or recomputes PPO's
rollout-pinned advantages and returns.

`rollout.batching.slice_batch()` accepts index lists or continuous slices; the
latter retain tensor views. Direct strategy replay uses
`slice_sequences=False` to keep its existing shared non-tensor metadata behavior.
Worker and global-window batches also select aligned Python sequences. Async
GRPO retains its weighted microbatch iterator and one update per round.

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
