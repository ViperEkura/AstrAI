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

BaseStrategy.prepare_batch() receives an offline batch or converts a
RolloutResult in online mode. compute_loss_output() returns loss and metrics.
The token-mean forward/reduce path supports context parallelism; each
algorithm retains its own reduction semantics. training_steps() and
training_updates() express microbatch or update shape while Trainer retains
commit ownership. ops.py also supplies chunked log-probability calculation
for the memory-sensitive path.

## Registry and capabilities

StrategyFactory stores a StrategyCapabilities declaration for each registered
name. TrainConfig validates the registry name and uses its capabilities for
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
accepted_options; internal StrategyFactory.create_checked rejects unknown
options. The legacy BaseFactory.create path warns when it discards an
argument for one minor version before becoming strict.

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
create_checked() to reject unknown options; the legacy BaseFactory.create()
warns for one minor version when it filters an option.
