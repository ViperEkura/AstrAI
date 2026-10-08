"""Small, synchronous GRPO runner with frozen task rewards and raw results."""

import argparse
import inspect
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
import yaml

from astrai.config import ConfigFactory, TrainConfig
from astrai.model import AutoRegressiveLM
from astrai.optim import OptimizerFactory
from astrai.serialization import adapt_config, load_json
from astrai.tokenize import AutoTokenizer
from astrai.trainer import Trainer
from astrai.trainer.callbacks.checkpoint import CheckpointCallback
from astrai.trainer.optional_extras import checkpoint_extras, restore_checkpoint_extras
from astrai.trainer.rollout import SamplingParams
from examples.rl_reward.data import (
    PromptDataset,
    collate_prompts,
    load_splits,
    sha256_file,
)
from examples.rl_reward.publication import (
    content_digest,
    dependency_versions,
    public_pretrained,
    public_recipe,
)
from examples.rl_reward.rewards import VERIFIER_VERSION, TaskReward


@dataclass
class Recipe:
    model_path: str
    model_repo: str
    model_revision: str
    train_file: str
    dev_file: str
    dataset_repo: str
    dataset_revision: str
    output_dir: str
    task: str = "countdown"
    optimizer: str = "muon_adamw"
    optimizer_kwargs: dict = field(
        default_factory=lambda: {"lr": 3e-4, "weight_decay": 0.1}
    )
    updates: int = 400
    seed: int = 3407
    batch_per_device: int = 1
    group_size: int = 8
    prompt_cap: int = 1024
    response_cap: int = 512
    eval_interval: int = 25
    eval_batch_size: int = 8
    checkpoint_interval: int = 100
    kl_coef: float = 0.01
    clip_eps: float = 0.2
    loss_aggregation: str = "token"
    dtype: str = "bfloat16"
    device_type: str = "cuda"
    reward_target: float | None = None
    noninferiority_margin: float = 0.02
    test_file: str | None = None
    save_token_traces: bool = True
    learner_microbatch_prompts: int | None = None
    overlap_collection: bool = False
    request_seeded_sampling: bool = False
    enable_thinking: bool | None = None

    def validate(self):
        for name in (
            "model_repo",
            "model_revision",
            "dataset_repo",
            "dataset_revision",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or value.lower() in {"main", "latest", "todo", "replace_me"}
            ):
                raise ValueError(f"{name} must identify a frozen source")
        if self.task not in {"countdown", "gsm8k"}:
            raise ValueError("task must be countdown or gsm8k")
        if self.dtype not in {"bfloat16", "float32"} or self.device_type not in {
            "cuda",
            "cpu",
        }:
            raise ValueError("supported dtype/device: bfloat16 or float32; cuda or cpu")
        if self.optimizer not in {"muon_adamw", "adamw"}:
            raise ValueError("optimizer must be muon_adamw or adamw")
        if self.loss_aggregation not in {"token", "sequence"}:
            raise ValueError("loss_aggregation must be token or sequence")
        for name in (
            "updates",
            "batch_per_device",
            "group_size",
            "prompt_cap",
            "response_cap",
            "eval_interval",
            "eval_batch_size",
            "checkpoint_interval",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.group_size < 2:
            raise ValueError("training group_size must be at least 2")
        if type(self.seed) is not int or type(self.save_token_traces) is not bool:
            raise ValueError("seed must be an integer and save_token_traces a boolean")
        if type(self.overlap_collection) is not bool:
            raise ValueError("overlap_collection must be a boolean")
        if type(self.request_seeded_sampling) is not bool:
            raise ValueError("request_seeded_sampling must be a boolean")
        if self.request_seeded_sampling and not 0 <= self.seed < 2**63:
            raise ValueError("request-local sampling requires seed in [0, 2**63-1]")
        if self.enable_thinking is not None and type(self.enable_thinking) is not bool:
            raise ValueError("enable_thinking must be a boolean or None")
        if self.learner_microbatch_prompts is not None and (
            type(self.learner_microbatch_prompts) is not int
            or self.learner_microbatch_prompts < 1
        ):
            raise ValueError("learner_microbatch_prompts must be positive or None")
        if not math.isfinite(self.kl_coef) or self.kl_coef < 0:
            raise ValueError("kl_coef must be finite and nonnegative")
        if not math.isfinite(self.clip_eps) or not 0 <= self.clip_eps < 1:
            raise ValueError("clip_eps must be in [0, 1)")
        if self.reward_target is not None and not 0 <= self.reward_target <= 1:
            raise ValueError("reward_target must be in [0, 1]")
        if not 0 <= self.noninferiority_margin <= 1:
            raise ValueError("noninferiority_margin must be in [0, 1]")
        if (
            not isinstance(self.optimizer_kwargs, dict)
            or "lr" not in self.optimizer_kwargs
        ):
            raise ValueError("optimizer_kwargs must declare lr")
        lr = self.optimizer_kwargs["lr"]
        if (
            isinstance(lr, bool)
            or not isinstance(lr, (int, float))
            or not math.isfinite(lr)
            or lr <= 0
        ):
            raise ValueError("optimizer lr must be finite and positive")
        for name in ("model_path", "train_file", "dev_file", "output_dir"):
            if not Path(getattr(self, name)).is_absolute():
                raise ValueError(f"{name} must be an absolute path")
        if self.test_file and not Path(self.test_file).is_absolute():
            raise ValueError("test_file must be an absolute path")


def command_output(args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def model_factory(config, dtype):
    return AutoRegressiveLM(ConfigFactory.load(config)).to(dtype=getattr(torch, dtype))


def configure_prompt(tokenizer, recipe):
    if recipe.enable_thinking is not None:
        template = getattr(tokenizer, "_chat_template", None)
        if template is None:
            raise ValueError("enable_thinking requires a loaded chat template")
        template.default_variables["enable_thinking"] = recipe.enable_thinking


def optimizer_factory(model, name, options):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **options)
    return OptimizerFactory.create(name, model, **options)


def resolve_optimizer(recipe):
    cls = (
        torch.optim.AdamW
        if recipe.optimizer == "adamw"
        else OptimizerFactory.get_component_class(recipe.optimizer)
    )
    parameters = inspect.signature(cls).parameters
    defaults = {
        name: parameter.default
        for name, parameter in parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    unknown = set(recipe.optimizer_kwargs) - set(defaults)
    if unknown:
        raise ValueError(f"unknown optimizer settings: {sorted(unknown)}")
    recipe.optimizer_kwargs = json.loads(
        json.dumps({**defaults, **recipe.optimizer_kwargs}, allow_nan=False)
    )


def scheduler_factory(optimizer):
    # Constant LR is explicit in the manifest; no implicit warmup recipe.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def save_runner_extra(context):
    extra = CheckpointCallback.save_extra(context)
    monitor = context.kwargs["reward_monitor"]
    extra["reward_runner"] = {
        "recipe": asdict(monitor.recipe),
        "data_hashes": monitor.data_hashes,
        "verifier_sha256": sha256_file(Path(__file__).with_name("rewards.py")),
        "rng_by_rank": context.kwargs["reward_rng_by_rank"],
        "elapsed_seconds": monitor.elapsed(),
    }
    return extra


def restore_runner_rng(states, rank, world_size):
    if not isinstance(states, list) or len(states) != world_size:
        raise ValueError(
            "resume requires one RNG snapshot per learner rank and the same topology"
        )
    restore_checkpoint_extras({"rng_state": states[rank]})


class RewardCheckpoint(CheckpointCallback):
    """Save after consumed_samples advances, at a complete round boundary."""

    def after_optimizer_step(self, context):
        pass

    def _save_checkpoint(self, context):
        if next(context.model.parameters()).is_cuda:
            torch.cuda.synchronize()
        local_rng = checkpoint_extras()["rng_state"]
        if dist.is_initialized():
            states = [None] * context.world_size
            dist.all_gather_object(states, local_rng)
        else:
            states = [local_rng]
        context.kwargs["reward_rng_by_rank"] = states
        super()._save_checkpoint(context)

    def on_batch_end(self, context):
        if context.optimizer_step - self.last_ckpt_step >= self.interval:
            self._save_checkpoint(context)


class Monitor:
    def __init__(self, recipe, splits, data_hashes, started):
        self.recipe, self.splits, self.data_hashes = recipe, splits, data_hashes
        self.started = started
        self.elapsed_before = 0.0
        self.phase = "train"
        self.raw = None
        self.collection_seconds = 0.0
        self.scoring_seconds = 0.0
        self.last_rewards = None
        self.round_started = None
        self.context = None
        self.attempt = uuid.uuid4().hex

    def elapsed(self):
        return self.elapsed_before + time.perf_counter() - self.started

    def sync(self):
        if self.recipe.device_type == "cuda":
            torch.cuda.synchronize()

    def write(self, kind, value):
        path = (
            Path(self.recipe.output_dir)
            / f"{kind}.rank{self.context.rank}.{self.attempt}.jsonl"
        )
        with path.open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")

    def memory(self, phase):
        if self.recipe.device_type == "cuda":
            self.write(
                "memory_by_rank",
                {
                    "phase": phase,
                    "rank": self.context.rank,
                    "policy_version": self.context.strategy.policy_version,
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "peak_allocated": torch.cuda.max_memory_allocated(),
                },
            )

    def on_train_begin(self, context):
        self.context = context
        context.kwargs["reward_monitor"] = self
        context.strategy._rollout_runner.reward_model.monitor = self
        # A resumed mid-epoch loader creates a new iterator. Its base-seed
        # draw must not advance the restored rollout sampling RNG.
        context.dataloader.generator = torch.Generator().manual_seed(
            self.recipe.seed + context.rank
        )
        if context.checkpoint:
            state = context.checkpoint.extra.get("reward_runner")
            if (
                state is None
                or state["recipe"] != asdict(self.recipe)
                or state["data_hashes"] != self.data_hashes
                or state.get("verifier_sha256")
                != sha256_file(Path(__file__).with_name("rewards.py"))
            ):
                raise ValueError(
                    "resume requires the original reward recipe and data hashes"
                )
            self.elapsed_before = state["elapsed_seconds"]
            # Builder-created reference/backend objects may consume random
            # draws after its initial restore; restore at the ready boundary.
            restore_runner_rng(
                state.get("rng_by_rank"), context.rank, context.world_size
            )
        else:
            random.seed(self.recipe.seed + context.rank)
            torch.manual_seed(self.recipe.seed + context.rank)
        if context.strategy.policy_version >= self.recipe.updates:
            raise ValueError("checkpoint already reached the frozen update budget")
        generator = context.strategy._rollout_runner.generator
        configure_prompt(generator.tokenizer, self.recipe)
        generate = generator.generate

        def timed_generate(*args, **kwargs):
            self.sync()
            start = time.perf_counter()
            raw = generate(*args, **kwargs)
            self.sync()
            self.collection_seconds = time.perf_counter() - start
            self.raw = raw
            self.memory("collection")
            return raw

        generator.generate = timed_generate
        runtime = {
            "rank": context.rank,
            "world_size": context.world_size,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "policy_version": context.strategy.policy_version,
            "dataloader_rng": "isolated CPU generator",
            "pretrained": public_pretrained(
                getattr(context, "pretrained_metadata", {})
            ),
        }
        h100 = torch.tensor(
            int(
                self.recipe.device_type == "cuda"
                and "H100" in torch.cuda.get_device_name(torch.cuda.current_device())
            ),
            device=next(context.model.parameters()).device,
        )
        if dist.is_initialized():
            dist.all_reduce(h100)
        runtime["h100_count"] = h100.item()
        self.write("runtime", runtime)
        self.memory("post_load")
        self.evaluate("dev")

    def on_batch_begin(self, context):
        self.phase = "train"
        self.sync()
        self.round_started = time.perf_counter()
        if self.recipe.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def record_rewards(self, records, responses, rewards, formats, scoring_seconds):
        self.last_rewards = rewards
        self.scoring_seconds = scoring_seconds
        raw = self.raw
        if raw is None or len(records) != raw.responses.shape[0]:
            raise ValueError("reward records are not bound to the current rollout")
        version = raw.policy_version
        groups = []
        for index, record in enumerate(records):
            group_id = f"{self.recipe.seed}:{version}:{record['id']}"
            groups.append(
                {
                    "prompt_id": record["id"],
                    "group_id": group_id,
                    "prompt_hash": record["prompt_hash"],
                    "split": record["split"],
                    "responses": [
                        {
                            "response_id": f"{group_id}:{response_index}",
                            "text": response,
                            "reward": rewards[index, response_index].item(),
                            "valid_format": formats[index][response_index],
                            "finish_reason": raw.finish_reasons[index][response_index],
                            "response_tokens": raw.response_mask[index, response_index]
                            .sum()
                            .item(),
                        }
                        for response_index, response in enumerate(responses[index])
                    ],
                }
            )
        self.write(
            "rollout_results" if self.phase == "train" else "eval_results",
            {
                "phase": self.phase,
                "seed": self.recipe.seed,
                "policy_version": version,
                "elapsed_seconds": self.elapsed(),
                "groups": groups,
            },
        )
        if self.recipe.save_token_traces:
            root = Path(self.recipe.output_dir) / "token_traces"
            root.mkdir(exist_ok=True)
            name = (
                f"{self.phase}.v{version}.rank{self.context.rank}.{uuid.uuid4().hex}.pt"
            )
            torch.save(
                {
                    "groups": [group["group_id"] for group in groups],
                    "policy_version": version,
                    "prompts": raw.prompts.cpu(),
                    "prompt_mask": raw.prompt_mask.cpu(),
                    "tokens": raw.responses.cpu(),
                    "response_mask": raw.response_mask.cpu(),
                    "policy_raw_logp": raw.logprobs_old.cpu(),
                    "rewards": rewards.cpu(),
                },
                root / name,
            )

    def evaluate(self, split):
        context = self.context
        self.phase = split
        self.sync()
        start = time.perf_counter()
        records = self.splits[split][context.rank :: context.world_size]
        count, reward_sum = 0, 0.0
        runner = context.strategy._rollout_runner
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_tokens=self.recipe.response_cap,
            group_size=1,
        )
        size = min(
            self.recipe.eval_batch_size,
            self.recipe.group_size * self.recipe.batch_per_device,
        )
        for begin in range(0, len(records), size):
            batch = collate_prompts(
                [record["messages"] for record in records[begin : begin + size]]
            )
            result = runner.evaluate(batch, params)
            reward_sum += result.rewards.sum().item()
            count += result.rewards.numel()
        totals = torch.tensor(
            [reward_sum, count],
            dtype=torch.float64,
            device=next(context.model.parameters()).device,
        )
        if dist.is_initialized():
            dist.all_reduce(totals)
        self.sync()
        self.write(
            "eval_metrics",
            {
                "split": split,
                "policy_version": context.strategy.policy_version,
                "reward_mean": (totals[0] / totals[1]).item(),
                "responses": totals[1].item(),
                "eval_seconds": time.perf_counter() - start,
                "elapsed_seconds": self.elapsed(),
            },
        )
        self.phase = "train"

    def on_batch_end(self, context):
        self.sync()
        self.write(
            "round_metrics",
            {
                "policy_version": context.strategy.policy_version,
                "optimizer_step": context.optimizer_step,
                "groups": self.raw.responses.shape[0],
                "valid_response_tokens": self.raw.response_mask.sum().item(),
                "collection_seconds": self.collection_seconds,
                "reward_cpu_seconds": self.scoring_seconds,
                "reward_mean": self.last_rewards.mean().item(),
                "nonzero_advantage_group_fraction": (
                    self.last_rewards.std(dim=-1, unbiased=False) > 0
                )
                .float()
                .mean()
                .item(),
                "round_seconds": time.perf_counter() - self.round_started,
                "elapsed_seconds": self.elapsed(),
                "metrics": context.metrics,
            },
        )
        self.memory("round_end")
        if context.optimizer_step % self.recipe.eval_interval == 0:
            self.evaluate("dev")
        if context.optimizer_step >= self.recipe.updates:
            if "test" in self.splits:
                self.evaluate("test")
            context.request_stop()

    def on_train_end(self, context):
        self.sync()
        self.write(
            "run_end",
            {
                "policy_version": context.strategy.policy_version,
                "optimizer_step": context.optimizer_step,
                "elapsed_seconds": self.elapsed(),
                "budget_completed": context.optimizer_step >= self.recipe.updates,
                "time_scope": "runner process; launcher cleanup and allocation accounting are external",
            },
        )


def build_training(recipe, splits, records_by_prompt, data_hashes, started):
    recipe.validate()
    resolve_optimizer(recipe)
    features = {}
    for field_name, value, required, needed in (
        (
            "rl_microbatch_prompts",
            recipe.learner_microbatch_prompts,
            "A2",
            recipe.learner_microbatch_prompts is not None,
        ),
        (
            "rollout_enable_overlap",
            recipe.overlap_collection,
            "R1",
            recipe.overlap_collection,
        ),
        (
            "rollout_seed",
            recipe.seed,
            "R1",
            recipe.request_seeded_sampling,
        ),
    ):
        if needed:
            if field_name not in TrainConfig.__dataclass_fields__:
                raise RuntimeError(
                    f"recipe requires the {required} implementation: {field_name}"
                )
            features[field_name] = value
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 1:
        raise ValueError("WORLD_SIZE must be positive")
    global_prompts = world * recipe.batch_per_device
    if len(splits["train"]) % global_prompts:
        raise ValueError(
            "train split must fill complete global batches; tail windows require A2"
        )
    config = ConfigFactory.load(
        adapt_config(
            load_json(Path(recipe.model_path) / "config.json"), recipe.model_path
        )
    )
    if recipe.prompt_cap + recipe.response_cap > config.max_position_embeddings:
        raise ValueError("prompt and response caps exceed the model context")
    monitor = Monitor(recipe, splits, data_hashes, started)
    train_config = TrainConfig(
        **features,
        model_fn=partial(model_factory, config.to_dict(), recipe.dtype),
        strategy="online_grpo",
        dataset=PromptDataset(splits["train"]),
        collate_fn=collate_prompts,
        optimizer_fn=partial(
            optimizer_factory, name=recipe.optimizer, options=recipe.optimizer_kwargs
        ),
        optimizer_name=recipe.optimizer,
        optimizer_hyperparameters=recipe.optimizer_kwargs,
        scheduler_fn=scheduler_factory,
        reward_model_fn=partial(TaskReward, records_by_prompt, recipe.task),
        n_epoch=math.ceil(recipe.updates / (len(splits["train"]) // global_prompts)),
        batch_per_device=recipe.batch_per_device,
        grad_accum_steps=1,
        dp_size=world,
        dp_mode="ddp" if world > 1 else "none",
        backend="nccl" if recipe.device_type == "cuda" else "gloo",
        device_type=recipe.device_type,
        random_seed=recipe.seed,
        ckpt_dir=str(Path(recipe.output_dir) / "checkpoints"),
        ckpt_interval=recipe.checkpoint_interval,
        rollout_interval=1,
        rollout_max_policy_lag=0,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=0,
        rollout_max_tokens=recipe.response_cap,
        rollout_pool_seq_len=recipe.prompt_cap + recipe.response_cap,
        strategy_kwargs={
            "group_size": recipe.group_size,
            "kl_coef": recipe.kl_coef,
            "clip_eps": recipe.clip_eps,
            "loss_aggregation": recipe.loss_aggregation,
        },
    )
    trainer = Trainer(train_config, callbacks=[monitor])
    trainer.callbacks = [
        RewardCheckpoint(
            callback.save_dir, callback.interval, save_extra_fn=save_runner_extra
        )
        if isinstance(callback, CheckpointCallback)
        else callback
        for callback in trainer.callbacks
    ]
    return trainer


def main():
    # Checkpoints retain exact private paths for resume. Never make their
    # directory or newly-created logs readable outside the launching user.
    os.umask(0o077)
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    recipe = Recipe(**yaml.safe_load(args.config.read_text()))
    recipe.validate()
    resolve_optimizer(recipe)
    model_dir = Path(recipe.model_path)
    if (model_dir / "hf_mapping.json").exists():
        from astrai.trainer import train_context

        if not hasattr(train_context, "prepare_pretrained_weights"):
            raise RuntimeError(
                "HF reward runs require the shared pretrained-loading correction (A1)"
            )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    configure_prompt(tokenizer, recipe)
    paths = {"train": recipe.train_file, "dev": recipe.dev_file}
    if recipe.test_file:
        paths["test"] = recipe.test_file
    splits, records_by_prompt = load_splits(
        paths, recipe.task, tokenizer, recipe.prompt_cap
    )
    data_hashes = {split: sha256_file(path) for split, path in paths.items()}
    root = Path(recipe.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    request_rng = "shared_torch_multinomial"
    if recipe.request_seeded_sampling:
        from astrai.inference.sampling_rng import REQUEST_RNG_VERSION

        request_rng = REQUEST_RNG_VERSION
    manifest = {
        "recipe": public_recipe(recipe),
        "dataset_sha256": data_hashes,
        "split_ids": {
            split: [record["id"] for record in records]
            for split, records in splits.items()
        },
        "verifier_version": VERIFIER_VERSION,
        "verifier_sha256": sha256_file(Path(__file__).with_name("rewards.py")),
        "git_head": command_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"]
        ),
        "git_diff_sha256": content_digest(
            command_output(
                [
                    "git",
                    "-C",
                    str(Path(__file__).resolve().parents[2]),
                    "diff",
                    "--binary",
                    "HEAD",
                ]
            )
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model_files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in model_dir.iterdir()
            if path.is_file() and (path.suffix in {".json", ".safetensors"})
        },
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "frequency_penalty": 0.0,
            "request_rng": request_rng,
            "request_seed": recipe.seed if recipe.request_seeded_sampling else None,
        },
        "prompt_template_variables": {"enable_thinking": recipe.enable_thinking},
        "tokenizer_stop_ids": tokenizer.stop_ids,
        "evaluation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "group_size": 1,
            "effective_batch_size": min(
                recipe.eval_batch_size, recipe.batch_per_device * recipe.group_size
            ),
        },
        "logprobs_old_semantics": "raw_model_logp; sampler_logq is not separately recorded",
        "learner": {
            "dp_size": int(os.environ.get("WORLD_SIZE", "1")),
            "cp_size": 1,
            "tp_size": 1,
            "grad_accum_steps": 1,
            "rl_update_epochs": 1,
            "rl_minibatch_prompts": None,
            "rl_microbatch_prompts": recipe.learner_microbatch_prompts,
        },
        "collector": {"enable_overlap": recipe.overlap_collection},
        "resume_checkpoint_present": args.resume is not None,
        "resume_checkpoint_manifest_sha256": sha256_file(args.resume / "manifest.json")
        if args.resume and (args.resume / "manifest.json").is_file()
        else None,
        "created_unix": time.time(),
    }
    name = (
        f"resume_manifest.rank{rank}.{uuid.uuid4().hex}.json"
        if args.resume
        else f"run_manifest.rank{rank}.json"
    )
    with (root / name).open("x") as stream:
        json.dump(manifest, stream, indent=2)
    (root / f"dependencies.rank{rank}.txt").write_text(
        "".join(
            f"{name}=={version}\n" for name, version in dependency_versions().items()
        )
    )
    trainer = build_training(recipe, splits, records_by_prompt, data_hashes, started)
    trainer.train(
        param_path=str(args.resume or model_dir), resume=args.resume is not None
    )


if __name__ == "__main__":
    main()
