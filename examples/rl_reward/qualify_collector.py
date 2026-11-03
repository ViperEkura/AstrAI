"""Official-model CUDA collector qualification with complete groups and EOS."""

import argparse
import gc
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from astrai.extension import op_backend
from astrai.inference.core.engine_core import EngineCore
from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer
from astrai.trainer.backend import ColocatedBackend
from astrai.trainer.rollout import RolloutGenerator, SamplingParams
from examples.rl_reward.data import collate_prompts, load_splits, sha256_file
from examples.rl_reward.publication import dependency_versions
from examples.rl_reward.rewards import TaskReward
from examples.rl_reward.run import Recipe, configure_prompt


def validate_episode(raw, concurrency, group, response_cap, stop_ids):
    if raw.responses.shape[:2] != (concurrency // group, group):
        raise ValueError("collector did not return every complete response group")
    mask = raw.response_mask
    lengths = mask.sum(-1)
    if not bool(((lengths > 0) & (lengths <= response_cap)).all()):
        raise ValueError("invalid response lengths")
    expected_mask = torch.arange(
        mask.shape[-1], device=mask.device
    ) < lengths.unsqueeze(-1)
    if not torch.equal(mask, expected_mask):
        raise ValueError(
            "response mask must retain every generated token including EOS"
        )
    if not bool(torch.isfinite(raw.logprobs_old[mask]).all()):
        raise ValueError("nonfinite raw policy logprobs")
    stops = 0
    for row in range(concurrency // group):
        for response in range(group):
            length = int(lengths[row, response])
            reason = raw.finish_reasons[row][response]
            if reason == "stop":
                stops += 1
                if int(raw.responses[row, response, length - 1]) not in stop_ids:
                    raise ValueError("stop response does not retain its terminal token")
            elif reason != "length" or length != response_cap:
                raise ValueError("unexpected or incomplete terminal response")
    if not stops or lengths.min() == lengths.max():
        raise ValueError("official qualification requires observed variable EOS")
    return {
        "responses": concurrency,
        "stop_responses": stops,
        "length_responses": concurrency - stops,
        "min_response_tokens": lengths.min().item(),
        "max_response_tokens": lengths.max().item(),
        "valid_response_tokens": mask.sum().item(),
    }


def assert_drained(engine):
    scheduler = engine.scheduler
    if (
        engine.pending
        or scheduler._planned
        or scheduler._pending_order
        or scheduler._ready
        or scheduler._states
        or scheduler._kv_manager.request_count
        or any(slot["in_use"] for slot in engine.model_runner._result_ring._slots)
    ):
        raise RuntimeError("collector did not drain execution, request and KV owners")


@torch.no_grad()
def replay_logprobs(model, raw):
    maximum = 0.0
    for row in range(min(raw.prompts.shape[0], 2)):
        prompt = raw.prompts[row][raw.prompt_mask[row]]
        for response in range(min(raw.responses.shape[1], 2)):
            mask = raw.response_mask[row, response]
            generated = raw.responses[row, response][mask]
            sequence = torch.cat((prompt, generated))
            logits = model(sequence[:-1].unsqueeze(0))["logits"][0]
            predicted = logits[len(prompt) - 1 :]
            expected = (
                predicted.float()
                .log_softmax(-1)
                .gather(-1, generated[:, None])
                .squeeze(-1)
            )
            actual = raw.logprobs_old[row, response][mask]
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.05)
            maximum = max(maximum, (actual - expected).abs().max().item())
    return maximum


def compare_pairs(baseline, candidate):
    for name in ("prompts", "prompt_mask", "responses", "response_mask"):
        torch.testing.assert_close(
            getattr(candidate, name), getattr(baseline, name), rtol=0, atol=0
        )
    torch.testing.assert_close(
        candidate.logprobs_old, baseline.logprobs_old, rtol=0.01, atol=0.05
    )
    if (
        baseline.finish_reasons != candidate.finish_reasons
        or baseline.policy_version != candidate.policy_version
    ):
        raise ValueError("paired collector terminal reasons or versions differ")


def public_groups(records, raw, scores):
    groups = []
    for row, record in enumerate(records):
        responses = []
        for index in range(raw.responses.shape[1]):
            mask = raw.response_mask[row, index]
            responses.append(
                {
                    "response_index": index,
                    "tokens": raw.responses[row, index][mask].tolist(),
                    "policy_raw_logp": raw.logprobs_old[row, index][mask].tolist(),
                    "text": raw.response_texts[row][index],
                    "finish_reason": raw.finish_reasons[row][index],
                    "reward": scores[row, index].item(),
                }
            )
        groups.append(
            {
                "prompt_id": record["id"],
                "prompt_hash": record["prompt_hash"],
                "responses": responses,
            }
        )
    return groups


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipe = Recipe(**yaml.safe_load(args.config.read_text()))
    recipe.validate()
    if args.output.exists() or not args.output.is_absolute():
        raise ValueError("qualification output must be a new absolute directory")
    if (
        recipe.device_type != "cuda"
        or recipe.group_size != 8
        or not recipe.request_seeded_sampling
    ):
        raise ValueError(
            "qualification requires CUDA, group size 8 and request-local sampling"
        )
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("qualification requires an assigned H100 test resource")
    if torch.__version__.split("+")[0] != "2.11.0":
        raise RuntimeError("qualification requires the pinned PyTorch version")
    args.output.mkdir(parents=True, mode=0o700)
    tokenizer = AutoTokenizer.from_pretrained(recipe.model_path)
    configure_prompt(tokenizer, recipe)
    if not tokenizer.stop_ids:
        raise ValueError("official tokenizer must declare terminal tokens")
    splits, records_by_prompt = load_splits(
        {"dev": recipe.dev_file}, recipe.task, tokenizer, recipe.prompt_cap
    )
    model = (
        AutoModel.from_pretrained(recipe.model_path, strict=True)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    model.requires_grad_(False)
    cases = []
    started = time.perf_counter()
    try:
        for concurrency in (32, 64, 128, 256):
            records = splits["dev"][: concurrency // 8]
            if len(records) != concurrency // 8:
                raise ValueError(
                    "frozen dev set is too small for requested concurrency"
                )
            batch = collate_prompts([r["messages"] for r in records])
            reward = TaskReward(records_by_prompt, recipe.task)
            for graphs in (False, True):
                paired = {}
                for overlap in (False, True):
                    with op_backend(attention="cuda"):
                        engine = EngineCore(
                            model,
                            tokenizer,
                            max_batch_size=concurrency,
                            max_seq_len=recipe.prompt_cap + recipe.response_cap,
                            enable_overlap=overlap,
                            enable_cuda_graph=graphs,
                        )
                    generator = RolloutGenerator(
                        ColocatedBackend(engine),
                        tokenizer,
                        SamplingParams(
                            group_size=8,
                            max_tokens=recipe.response_cap,
                            temperature=1,
                            top_p=1,
                            top_k=0,
                            seed=recipe.seed,
                        ),
                    )
                    depths = []
                    submit = engine.submit

                    def observe(plan):
                        depths.append(len(engine.pending))
                        return submit(plan)

                    engine.submit = observe
                    try:
                        generator.generate(batch)
                        assert_drained(engine)
                        for seed in (3407, 3408, 3409):
                            depths.clear()
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            before = time.perf_counter()
                            raw = generator.generate(
                                batch, replace(generator.params, seed=seed)
                            )
                            torch.cuda.synchronize()
                            wall = time.perf_counter() - before
                            peak_allocated = torch.cuda.max_memory_allocated()
                            peak_reserved = torch.cuda.max_memory_reserved()
                            metrics = validate_episode(
                                raw,
                                concurrency,
                                8,
                                recipe.response_cap,
                                set(tokenizer.stop_ids),
                            )
                            assert_drained(engine)
                            if (1 in depths) != overlap:
                                raise ValueError(
                                    "collector did not dispatch the declared overlap pipeline"
                                )
                            scores = reward.score(raw.prompt_texts, raw.response_texts)
                            logp_error = replay_logprobs(model, raw)
                            if overlap:
                                baseline, baseline_scores = paired[seed]
                                compare_pairs(baseline, raw)
                                torch.testing.assert_close(
                                    scores, baseline_scores, rtol=0, atol=0
                                )
                            else:
                                paired[seed] = (raw, scores)
                            captured = len(engine.model_runner._graph_ctx._graphs)
                            if graphs and not captured:
                                raise ValueError(
                                    "requested CUDA graphs were not captured"
                                )
                            case = {
                                "state": "PASS_CASE",
                                "concurrency": concurrency,
                                "group_size": 8,
                                "graphs": graphs,
                                "captured_graphs": captured,
                                "overlap": overlap,
                                "seed": seed,
                                "policy_version": raw.policy_version,
                                "collection_seconds": wall,
                                "episode": metrics,
                                "raw_logprob_replay_max_error": logp_error,
                                "peak_allocated_bytes": peak_allocated,
                                "peak_reserved_bytes": peak_reserved,
                                "memory_scope": "timed collection; model and paired rollout traces resident",
                                "paired_equality_validated": overlap,
                                "pending_depth_before_submit": depths,
                                "groups": public_groups(records, raw, scores),
                            }
                            with (args.output / "cases.jsonl").open("a") as stream:
                                stream.write(json.dumps(case, allow_nan=False) + "\n")
                            cases.append(
                                {
                                    key: value
                                    for key, value in case.items()
                                    if key != "groups"
                                }
                            )
                    finally:
                        engine.stop()
                        del generator, engine
                        gc.collect()
                        torch.cuda.empty_cache()
        summary = {
            "state": "PASS_OFFICIAL_COLLECTOR_QUALIFICATION",
            "h100_count": 1,
            "cuda": torch.version.cuda,
            "dependency_versions": dependency_versions(),
            "model_repo": recipe.model_repo,
            "model_revision": recipe.model_revision,
            "dev_file_sha256": sha256_file(recipe.dev_file),
            "group_size": 8,
            "temperature": 1,
            "top_p": 1,
            "top_k": 0,
            "frequency_penalty": 0,
            "eos_enabled": True,
            "case_count": len(cases),
            "cases": cases,
            "qualification_seconds": time.perf_counter() - started,
            "full_rl_learning": "NOT_TESTED_BY_THIS_CHECK",
        }
        (args.output / "collector-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(
            json.dumps(
                {"state": summary["state"], "case_count": len(cases), "h100_count": 1}
            )
        )
    except BaseException as error:
        (args.output / "failure.json").write_text(
            json.dumps(
                {
                    "state": "FAIL_OR_INCOMPLETE",
                    "failure_type": type(error).__name__,
                    "completed_cases": len(cases),
                }
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
