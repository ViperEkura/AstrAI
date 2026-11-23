from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from astrai.config import AutoRegressiveLMConfig, ComponentSpec
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.parallel.pipeline import DecoderStage, PipelineUpdate
from astrai.trainer.components import build_optimizer_component


def _compare(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "pp"))
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
        reference = AutoRegressiveLM(cfg)
        spec = ComponentSpec(type="muon_adamw", params={"lr": 2e-4})
        ref_opt = build_optimizer_component(spec, reference)
        stage = DecoderStage(deepcopy(reference), [0, 2, 3], rank % 2)
        update = PipelineUpdate(stage, mesh, 1, 8, 2)
        sharded = ComponentSpec(
            type="muon_adamw", params={"lr": 2e-4, "shard_state": True}
        )
        opt = build_optimizer_component(
            sharded, stage, process_group=mesh["dp"].get_group()
        )
        for _ in range(2):
            tokens = torch.randint(64, (4, 9))
            inputs, labels = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
            ref_opt.zero_grad()
            loss = F.cross_entropy(
                reference(inputs)["logits"].flatten(0, 1), labels.flatten()
            )
            loss.backward()
            opt.zero_grad()
            part = slice(2 * (rank // 2), 2 * (rank // 2) + 2)
            measured = update.backward(inputs[part], labels[part])
            torch.testing.assert_close(measured, loss)
            full = dict(reference.named_parameters())
            for name, param in stage.named_parameters():
                torch.testing.assert_close(
                    param.main_grad, full[name].grad, atol=2e-6, rtol=1e-4
                )
            norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
            torch.testing.assert_close(update.clip_grad_norm(0.3), norm)
            opt.step()
            ref_opt.step()
            for name, param in stage.named_parameters():
                torch.testing.assert_close(param, full[name])
    finally:
        dist.destroy_process_group()


def test_pipeline_matches_full_model(tmp_path):
    mp.spawn(_compare, args=((tmp_path / "rdzv").as_uri(),), nprocs=4, join=True)
