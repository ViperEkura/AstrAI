"""Explicit repair of progress metadata in a confirmed legacy checkpoint."""

from pathlib import Path

import click

from astrai.serialization import Checkpoint
from astrai.trainer.callbacks.checkpoint import _copy_tokenizer_files
from astrai.trainer.train_context import repair_legacy_training_progress


@click.command(name="repair-progress")
@click.argument("source", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
def repair_progress_command(source: Path, output: Path) -> None:
    """Write a repaired copy of a confirmed legacy SOURCE checkpoint to OUTPUT.

    Model weights and optimizer/scheduler states are preserved. The source is
    not modified, and OUTPUT must be a new directory outside SOURCE.
    """
    source, output = source.resolve(), output.resolve()
    if output.exists() or source in output.parents:
        raise click.UsageError("OUTPUT must be a new directory outside SOURCE")
    try:
        checkpoint = Checkpoint.load(source)
        step, cursor = repair_legacy_training_progress(checkpoint)
        checkpoint.meta["optimizer_step"] = step
        checkpoint.meta["consumed_samples"] = cursor
        checkpoint.consumed_samples = cursor
        checkpoint.save(output)
        _copy_tokenizer_files(str(source), str(output))
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Repaired checkpoint: {output}")
    click.echo(f"optimizer_step={step}, consumed_samples={cursor}")


if __name__ == "__main__":
    repair_progress_command()
