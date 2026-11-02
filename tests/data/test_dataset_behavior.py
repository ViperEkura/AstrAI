import pytest
import torch

from astrai.dataset.dataset import (
    DatasetFactory,
    grpo_collate_fn,
)
from astrai.serialization import save_bin


@pytest.mark.parametrize("token_count,stride", [(200, 32), (200, 64), (30, 64)])
def test_dataset_windows_follow_stride_and_keep_shifted_targets(
    base_test_env, token_count, stride
):
    source = torch.arange(token_count)
    save_bin(base_test_env["test_dir"], {"sequence": [source]})
    dataset = DatasetFactory.load(
        "seq", base_test_env["test_dir"], window_size=64, stride=stride
    )
    assert dataset.token_count == token_count
    assert len(dataset) == max(0, (token_count - 1 - 64) // stride + 1)
    for index in range(len(dataset)):
        begin = index * stride
        torch.testing.assert_close(
            dataset[index]["input_ids"], source[begin : begin + 64]
        )
        torch.testing.assert_close(
            dataset[index]["target_ids"], source[begin + 1 : begin + 65]
        )


def test_grpo_collate_variable_lengths():
    batch = [
        {
            "prompts": torch.tensor([1, 2, 3]),
            "responses": [torch.tensor([4, 5]), torch.tensor([6, 7, 8, 9])],
            "masks": [torch.tensor([1, 1]), torch.tensor([1, 1, 1, 1])],
            "rewards": torch.tensor([0.9, 0.1]),
        },
        {
            "prompts": torch.tensor([10, 11]),
            "responses": [torch.tensor([12]), torch.tensor([13, 14, 15])],
            "masks": [torch.tensor([1]), torch.tensor([1, 1, 1])],
            "rewards": torch.tensor([0.5, 0.5]),
        },
    ]

    result = grpo_collate_fn(batch)

    assert result["prompts"].shape == (2, 3)  # B=2, P_max=3
    assert result["responses"].shape == (2, 2, 4)  # B=2, G=2, R_max=4
    assert result["masks"].shape == (2, 2, 4)
    assert result["rewards"].shape == (2, 2)

    # Prompts are left-padded so each response follows its real prompt tokens.
    assert torch.equal(result["prompts"][1], torch.tensor([0, 10, 11]))
    assert torch.equal(result["prompt_mask"][1], torch.tensor([False, True, True]))

    # Check response content: item 0, response 0 is [4,5] padded to 4
    assert result["responses"][0, 0, 0] == 4
    assert result["responses"][0, 0, 1] == 5
    assert result["responses"][0, 0, 2] == 0  # padded
    assert not result["masks"][0, 0, 2]  # padded

    # Check response content: item 0, response 1 is [6,7,8,9] no padding
    assert result["responses"][0, 1, 3] == 9
    assert result["masks"][0, 1, 3]
