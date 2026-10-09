import torch
from torch.utils.data import Dataset


class RandomTokenDataset(Dataset):
    """Random token dataset combining all test dataset variants.

    Parameters
    ----------
    length : int or None
        Fixed length, or ``None`` for a random length in [100, 200).
    max_length : int
        Sequence length per sample.
    vocab_size : int
        Upper bound for random token ids.
    with_loss_mask : bool
        Include a ``loss_mask`` key in each sample.
    stop_after : int or None
        Raise ``RuntimeError`` after this many samples (for early-stopping tests).
    """

    def __init__(
        self,
        length=100,
        max_length=64,
        vocab_size=1000,
        *,
        with_loss_mask=False,
        stop_after=None,
    ):
        self.length = (
            length if length is not None else int(torch.randint(100, 200, (1,)).item())
        )
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.with_loss_mask = with_loss_mask
        self.stop_after = stop_after
        self._count = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.stop_after is not None:
            self._count += 1
            if self._count == self.stop_after:
                raise RuntimeError("Simulated early stopping")

        item = {
            "input_ids": torch.randint(0, self.vocab_size, (self.max_length,)),
            "target_ids": torch.randint(0, self.vocab_size, (self.max_length,)),
        }
        if self.with_loss_mask:
            item["loss_mask"] = torch.randint(0, 1, (self.max_length,))
        return item


class FakeExecutor:
    """Executor stub tracking ``sync_gradients`` and providing ``unwrap_model``."""

    use_distributed = False

    def __init__(self, sync_gradients=True):
        self._sync_gradients = sync_gradients

    @property
    def sync_gradients(self):
        return self._sync_gradients

    def unwrap_model(self, model):
        return model.state_dict()
