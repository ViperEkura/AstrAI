# Task reward GRPO example

This example binds deterministic task rewards to AstrAI's existing native
rollout and `Trainer`. It writes resolved configuration, dataset/model hashes,
complete response groups, raw token traces, reward evaluations, phase times,
memory samples and resumable checkpoints. It supports synchronous GRPO with one
update per collected round, `grad_accum_steps=1`, and colocated rollout.

Install the repository's pinned dependencies with `pip install -e '.[dev]'`.
Prepare local model weights, tokenizer and, for HF checkpoints, a model-owned
`hf_mapping.json`. HF runs require the shared training/inference loader
correction (A1). Multi-rank learning/performance comparisons require the global
update normalization correction (A2); this example does not supply that change.
FSDP, role-separated rollout, adaptive loops and new model adapters are separate
feature gates.

After integrating A2, `learner_microbatch_prompts` selects complete prompt
groups per forward/backward without changing the collected global batch or
number of optimizer updates. After integrating R1, `overlap_collection: true`
selects the native submit/commit collector; `false` is the synchronous control.
These choices are recorded in the frozen recipe and manifest. Requesting a
feature before its implementation is present fails at startup. Keep these
settings fixed across resume, and compare performance arms with the same
global objective and workload.

After integrating R1, `request_seeded_sampling: true` binds sampling to the
seed, tokenized prompt, response index and policy version. Each output position
uses its own draw, so variable EOS, row reordering and discarded overlap work
do not advance another request's RNG. Both performance arms must use the same
sampler. The default remains shared multinomial sampling. `enable_thinking`
sets that chat-template variable consistently for data validation and rollout;
leave it unset for templates without that option. HF terminal tokens require
the A1 tokenizer correction. EOS remains enabled in formal qualification.

## Data and rewards

Provide immutable JSONL train/dev files with globally unique `id` values. An
optional `test_file` is evaluated only at the final update. Prepare the dataset
from a pinned revision outside this runner; downloading and split selection are
explicit preparation steps. The runner rejects duplicate tasks across splits,
invalid labels and prompts exceeding the cap. Train size must be divisible by
`WORLD_SIZE * batch_per_device` until tail update windows are supported.

`python -m examples.rl_reward.prepare_data --help` prepares splits on the
assigned CPU test machine from local Parquet files with declared SHA256 hashes
and immutable dataset revision. It requires `pyarrow==21.0.0`. Countdown
permutations and GSM8K whitespace variants are deduplicated globally before
selection; conflicting labels fail. A fixed hash order selects exact declared
sizes, and the original GSM8K test source remains held out. Preparation checks
every selected native chat prompt against the frozen cap and emits split hashes
and IDs. It fails if deduplication leaves too few records.

Countdown record:

```json
{"id":"train-0001","numbers":[7,3,2],"target":5}
```

The prompt asks for `<answer>(7+3)/2</answer>`. Every supplied integer, including
duplicate occurrences, must be used once. Allowed operations are `+`, `-`, `*`,
`/`, parentheses and unary signs. The verifier walks a bounded arithmetic AST
and evaluates exact rational values. It rejects floats, powers, floor division,
calls and attribute access. Every invalid or incorrect response receives zero;
no response is filtered from its group. This uses the number-use rule in the
[TinyZero verifier](https://github.com/Jiayi-Pan/TinyZero/blob/main/verl/utils/reward_score/countdown.py)
with an AST evaluator and binary correctness reward. It differs from that
verifier's formatting credit and floating-point tolerance, and is not a
score-centering paper reproduction. The corresponding
[Countdown dataset](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
uses `nums`; rename that field to `numbers` during fixed split preparation.

GSM8K record:

```json
{"id":"train-0001","question":"What is 3 plus 4?","answer":"3 + 4 = 7\n#### 7"}
```

Set `task: gsm8k`. Final answers are extracted from the last `####` line or
`\boxed{...}` and compared as exact numbers; commas, decimals and integer
fractions are supported. This is a numeric-answer verifier, not general symbolic
MATH equivalence. Invalid output scores zero. Ground-truth labels stay out of
model prompts. Rendered chat-template strings bind rewards to their records,
and prompt/dedup hashes are recorded.

## Run and resume

Copy `countdown.yaml`, fill absolute paths and immutable model/data identities,
and use a new output directory. Unknown YAML keys fail at startup. The supplied
400-update budget is a template; first use a 3–10 update capability run and an
independent reward pilot. Check the fraction of groups with nonzero advantages
before freezing a learning recipe. Set `reward_target` before comparing
time-to-target; the example records it but does not manufacture a crossing or a
quality decision.

```bash
python examples/rl_reward/run.py --config /absolute/path/to/recipe.yaml

# One learner process per GPU, with WORLD_SIZE used as TrainConfig.dp_size.
# For multiple nodes, use the allocation's rendezvous/torchrun setup.
torchrun --standalone --nproc_per_node=2 examples/rl_reward/run.py \
  --config /absolute/path/to/recipe.yaml

# Start a new process with the same frozen recipe and output directory.
python examples/rl_reward/run.py --config /absolute/path/to/recipe.yaml \
  --resume /absolute/path/to/checkpoints/epoch_0_step_100
```

Run tests and capability checks on the assigned test machines. CUDA runs require
an assigned GPU; the CPU smoke uses `device_type: cpu`, `dtype: float32`,
`optimizer: adamw`, a tiny native model, and short caps. CPU smoke proves path
execution and checkpoint behavior, not model learning or H100 performance.

Training uses temperature 1, top-p 1, top-k 0, no penalties, lag 0, one RL epoch
and no learner minibatch subdivision. Dev evaluation is greedy with group size
1, at step 0 and every `eval_interval` updates; test evaluation occurs at the
final update. `logprobs_old` is recorded as raw model log-probability; actual
sampler log-q is not a separate measured field. Optimizer defaults are resolved
from the installed implementation and recorded; LR is constant.

Checkpoints are saved at complete-round boundaries after the data cursor
advances. They retain actor, optimizer/scheduler, the original frozen reference,
registered RNG state and runner recipe/data identity. Resume checks the frozen
recipe and data hashes and restores RNG after reference/backend construction.
Each learner's RNG is gathered into the checkpoint and restored to that same
rank, and topology changes fail explicitly. The data-loader iterator has its
own generator so creating a fresh iterator does not advance rollout RNG; this
separates its [base-seed draw](https://docs.pytorch.org/docs/main/data.html#randomness-in-multi-process-data-loading)
from policy sampling. Verifier source hashes are also checked on resume.
Synchronous batch request IDs also preserve response-row order through the
prefill sort; independent random IDs would reassign the restored sampling
stream even when the RNG state itself matches.
An attempt-specific manifest and JSONL files preserve earlier results. Repeated
evaluation after resume has the same seed/version/prompt/response identity and
must be deduplicated when computing intervals.

## Results and timing

`run_manifest.rank*.json` records declared revisions, actual model/data hashes,
tokenizer/mapping files, source SHA/diff hash, sampling and learner settings.
Private paths are redacted. Runtime environment records include only Python,
necessary dependency/CUDA versions and H100 count; they omit host, scheduler,
hardware identity and memory capacity. Pretrained metadata uses a closed
allowlist of hashes/counts. Dependency files list versions without editable
paths or direct URLs. Exact recipe paths remain in private checkpoints for
resume. New outputs are created with user-only permissions. Publish only
reviewed experiment records, never raw launcher stderr or checkpoint metadata.

`rollout_results` and `eval_results` retain every response score, format status,
finish reason and logical ID. Optional `token_traces/*.pt` retain token IDs,
prompt/response masks, raw policy log-probs and rewards. `round_metrics` includes
reward/advantage statistics, valid response tokens, native collection wall,
CPU verifier wall and complete-round wall. Dev metrics reduce over all ranks;
every rank enters that collective even when its dev shard is empty. The eval
batch is bounded by the native scheduler's training capacity.

GPU phase timers synchronize the assigned device; memory observations cover
post-load, collection and round end. The round includes learner updates and
publication, result recording, and any periodic checkpoint callback. It does
not separately attribute forward/backward/optimizer kernel time. Runner elapsed
time includes initialization and evaluation; resume accumulates time retained
through its checkpoint. A launcher/controller must separately account for lost
work after a checkpoint, cleanup, scheduler allocation wall and every allocated
GPU when computing GPU-hours. These logs alone are not allocation-cost or
end-to-end speedup evidence.

Use the raw dev points for reward-vs-step and reward-vs-wall-time plots, retaining
all seeds, truncation, failures and unreached targets. Fixed-work performance
arms must share correctness fixes, data, model, optimizer, sampling, trace
logging and resource accounting. A multi-seed quality/performance result remains
an experimental deliverable beyond this runner.
