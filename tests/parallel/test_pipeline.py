from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from astrai.config import AutoRegressiveLMConfig, ComponentSpec
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.parallel.pipeline import DecoderStage, PipelineExecutor, PipelineUpdate
from astrai.parallel.setup import find_free_port
from astrai.trainer.components import build_optimizer_component
from tests.support.capabilities import requires_nccl


def _compare(rank, rendezvous, device_type):
    torch.set_num_threads(1)
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    device = (
        torch.device(device_type, rank)
        if device_type == "cuda"
        else torch.device("cpu")
    )
    backend = "nccl" if device_type == "cuda" else "gloo"
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=4)
    try:
        mesh = init_device_mesh(device_type, (2, 2), mesh_dim_names=("dp", "pp"))
        torch.manual_seed(82)
        cfg = AutoRegressiveLMConfig.from_dict(
            dict(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=3,
                max_position_embeddings=32,
                rms_norm_eps=1e-5,
                tie_word_embeddings=False,
                attention=dict(default_type="gqa", num_heads=2, num_kv_heads=1),
            )
        )
        reference = AutoRegressiveLM(cfg).to(device)
        spec = ComponentSpec(type="muon_adamw", params={"lr": 2e-4})
        ref_opt = build_optimizer_component(spec, reference)
        stage = DecoderStage(
            deepcopy(reference), [0, 2, 3], rank % 2, checkpoint_layers=1
        )
        update = PipelineUpdate(stage, mesh, 1, 8, 2)
        sharded = ComponentSpec(
            type="muon_adamw", params={"lr": 2e-4, "shard_state": True}
        )
        opt = build_optimizer_component(
            sharded, stage, process_group=mesh["dp"].get_group()
        )
        executor = PipelineExecutor(
            SimpleNamespace(
                grad_accum_steps=2,
                precision=SimpleNamespace(gradient_dtype="float32"),
                pp_size=2,
                strategy_kwargs={},
            ),
            SimpleNamespace(dp_rank=rank // 2, pp_rank=rank % 2, mesh=mesh),
        )
        for _ in range(2):
            tokens = torch.randint(64, (4, 9), device=device)
            inputs, labels = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
            ref_opt.zero_grad()
            loss = F.cross_entropy(
                reference(inputs)["logits"].flatten(0, 1), labels.flatten()
            )
            # Validation must not depend on constructing a training schedule.
            with torch.no_grad():
                measured_val = executor.validation_loss(
                    stage, {"input_ids": inputs, "target_ids": labels}
                )
            torch.testing.assert_close(measured_val, loss)
            assert executor.pipeline is None
            loss.backward()
            opt.zero_grad()
            part = slice(2 * (rank // 2), 2 * (rank // 2) + 2)
            measured = update.backward(inputs[part], labels[part])
            torch.testing.assert_close(measured, loss)
            full = dict(reference.named_parameters())
            for name, param in stage.named_parameters():
                torch.testing.assert_close(
                    param.main_grad,
                    full[name.replace("._checkpoint_wrapped_module", "")].grad,
                    atol=2e-6,
                    rtol=1e-4,
                )
            norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
            torch.testing.assert_close(update.clip_grad_norm(0.3), norm)
            opt.step()
            ref_opt.step()
            for name, param in stage.named_parameters():
                torch.testing.assert_close(
                    param, full[name.replace("._checkpoint_wrapped_module", "")]
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "device_type",
    ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_nccl(4)])],
)
def test_pipeline_matches_full_model(tmp_path, device_type):
    mp.spawn(
        _compare, args=((tmp_path / "rdzv").as_uri(), device_type), nprocs=4, join=True
    )


@pytest.mark.parametrize(
    "device_type",
    ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_nccl(4)])],
)
def test_train_cli_resumes_pipeline_window(tmp_path, device_type):
    import json
    import subprocess
    import sys

    import numpy as np
    import yaml
    from safetensors.torch import load_file
    from torch.utils._pytree import tree_flatten

    cfg = AutoRegressiveLMConfig.from_dict(
        dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            max_position_embeddings=16,
            rms_norm_eps=1e-5,
            tie_word_embeddings=False,
            attention=dict(default_type="gqa", num_heads=2, num_kv_heads=1),
        )
    )
    initial, data = tmp_path / "initial", tmp_path / "data"
    AutoRegressiveLM(cfg).save_pretrained(initial)
    data.mkdir()
    seq = np.arange(41, dtype=np.int32)
    seq.tofile(data / "sequence.bin")
    (data / "meta.json").write_text(
        json.dumps({"sequence": {"shape": [41], "dtype": "int32"}})
    )
    config = {
        "data": {
            "train_type": "seq",
            "data_root_path": str(data),
            "window_size": 8,
            "num_workers": 0,
        },
        "model": {"param_path": str(initial)},
        "parallel": {
            "pp_size": 2,
            "dp_size": 2 if device_type == "cuda" else 1,
            "dp_mode": "ddp" if device_type == "cuda" else "none",
            "backend": "nccl" if device_type == "cuda" else "gloo",
            "device_type": device_type,
            "pp_checkpoint_counts": [1, 0],
        },
        "training": {
            "grad_accum_steps": 2 if device_type == "cuda" else 4,
            "batch_per_device": 1,
            "n_epoch": 1,
            "epoch_tail_policy": "flush",
        },
        "precision": {
            "compute_dtype": "bfloat16" if device_type == "cuda" else "float32"
        },
        "ckpt": {"ckpt_interval": 1, "ckpt_dir": str(tmp_path / "first")},
    }

    def run(*args):
        config["parallel"]["master_port"] = find_free_port()
        path = tmp_path / "train.yaml"
        path.write_text(yaml.safe_dump(config))
        subprocess.run(
            [sys.executable, "-m", "scripts.tools.train", "--config", str(path), *args],
            check=True,
            timeout=120,
            capture_output=True,
            text=True,
        )

    run()
    config["model"]["param_path"] = str(tmp_path / "first" / "epoch_0_step_1")
    config["ckpt"]["ckpt_dir"] = str(tmp_path / "resume")
    run("--resume")
    paths = [tmp_path / p / "epoch_0_step_2" for p in ("first", "resume")]
    for path in paths:
        assert json.loads((path / "meta.json").read_text())["consumed_samples"] == 5
    a, b = [load_file(path / "model.safetensors") for path in paths]
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)

    for filename in ("optimizer.pt", "scheduler.pt"):
        saved = [
            torch.load(path / filename, map_location="cpu", weights_only=True)
            for path in paths
        ]
        (left, left_spec), (right, right_spec) = [
            tree_flatten(state) for state in saved
        ]
        assert left_spec == right_spec
        for before, after in zip(left, right):
            if isinstance(before, torch.Tensor):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            else:
                assert before == after
