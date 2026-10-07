"""CUDA Graph capture must use a stream on each replica's own device."""

import torch

from astrai.inference.worker.graph import CUDAGraphRunner
from tests.conftest import skip_lt2_cuda


@skip_lt2_cuda
def test_graph_capture_and_replay_on_multiple_devices():
    for device_index in range(1, min(torch.cuda.device_count(), 5)):
        device = torch.device("cuda", device_index)
        with torch.cuda.device(device):
            inputs = torch.ones((128, 128), device=device)
            weights = torch.full_like(inputs, 0.25)
            runner = CUDAGraphRunner(enabled=True)

            def forward(input_ids, weight):
                return {"logits": input_ids @ weight}

            args = {"input_ids": inputs, "weight": weights}
            runner.forward(forward, key=(128,), **args)
            runner.forward(forward, key=(128,), **args)
            assert runner.has_graph((128,))

            inputs.fill_(2)
            replayed = runner.forward(forward, key=(128,), **args)["logits"]
            torch.cuda.synchronize(device)
            torch.testing.assert_close(replayed, inputs @ weights, rtol=0, atol=0)

            runner.freeze_captures()
            eager = runner.forward(forward, key=(256,), **args)["logits"]
            torch.testing.assert_close(eager, inputs @ weights, rtol=0, atol=0)
            assert not runner.has_graph((256,))
