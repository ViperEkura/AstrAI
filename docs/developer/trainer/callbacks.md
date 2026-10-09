# Callbacks and checkpoint state

Callbacks observe a TrainContext at lifecycle boundaries. Trainer dispatches
epoch, batch and optimizer hooks. TrainSession dispatches startup and teardown
hooks and makes a best-effort cleanup pass when startup or training fails.

## Source modules

| Module | Responsibility |
| --- | --- |
| callbacks/base.py | TrainCallback protocol and callback registry |
| callbacks/checkpoint.py | Periodic, final and emergency checkpoint writing |
| callbacks/metrics.py | Training metrics, validation and event output |
| callbacks/metric_util.py | Metric accessors, gradient norm and GradSNRTracker |
| callbacks/optimization.py | Gradient clipping and activation checkpointing |
| callbacks/progress.py | Epoch progress display |
| optional_extras.py | Global RNG/FP8 and strategy-owned checkpoint extras |

## Event order

on_train_begin runs after context construction. Each epoch gets
on_epoch_begin/on_epoch_end. Batches get on_batch_begin/on_batch_end; a
committed optimizer update gets before_optimizer_step followed by
after_optimizer_step. on_error reports a stop or exception; on_train_end
runs during session cleanup. A failure before a context exists is handled
by TrainContextBuilder and does not dispatch context callbacks.

Gradient clipping acts before the optimizer step. Activation checkpointing is
enabled at startup and disabled at teardown. Progress and metrics read
completed context state; neither commits optimizer updates.

## Persistence

CheckpointCallback writes model, optimizer, scheduler, settings metadata and
registered extras at the configured interval. It also handles final and
emergency saves. It copies tokenizer files from the launch checkpoint when
available so an online checkpoint can resume independently.

optional_extras.py records Python, PyTorch, CUDA and NumPy RNG states and FP8
state through global providers. ComponentExtra entries describe strategy-owned
state such as a critic or frozen reference; the builder restores each after
its owner exists. Unknown optional keys are ignored on restore. Metadata
comes from TrainConfig.to_dict(), which reads settings without copying
runtime datasets.
