# AstrAI Architecture

## Contents

- [Class Diagram](#class-diagram) — Full Mermaid class diagram across 10+ namespaces
- [Module Overview](#module-overview) — Component inventory per module
- [Design Patterns](#design-patterns) — 16 documented patterns with classes
- [Core Relationships](#core-relationships) — 11 key inter-component relationships

## Class Diagram

```mermaid
classDiagram
    namespace config {
        class BaseConfig {
            +to_dict() Dict
            +from_dict(d) Self
            +from_file(path) Self
            +to_file(path)
        }

        class BaseModelConfig {
            +Optional[str] model_type
            +float neftune_alpha
            +from_file(config_path) Self
            +to_file(config_path)
        }

        class AutoRegressiveLMConfig {
            +Optional[int] vocab_size
            +Optional[int] hidden_size
            +Optional[int] num_hidden_layers
            +Optional[float] rms_norm_eps
            +Optional[int] intermediate_size
            +Optional[bool] tie_word_embeddings
            +Optional[dict] rope_scaling
            +Optional[int] max_position_embeddings
            +Optional[float] rope_theta
            +str attn_type
            +Optional[int] num_attention_heads
            +Optional[int] num_key_value_heads
            +Optional[bool] use_qk_norm
            +Optional[bool] use_gated_attention
            +Optional[int] kv_lora_rank
            +Optional[int] qk_nope_head_dim
            +Optional[int] qk_rope_head_dim
            +str ffn_type
            +Optional[int] n_routed_experts
            +Optional[int] n_shared_experts
            +Optional[int] n_activated_experts
            +Optional[str] topk_method
            +Optional[int] moe_intermediate_size
            +Optional[int] shared_expert_intermediate_size
            +bool norm_topk_prob
            +int decoder_sparse_step
            +Optional[List[int]] mlp_only_layers
        }

        class EncoderConfig {
            +Optional[int] vocab_size
            +Optional[int] hidden_size
            +Optional[int] num_hidden_layers
            +Optional[float] rms_norm_eps
            +Optional[int] intermediate_size
            +Optional[int] max_position_embeddings
            +Optional[float] rope_theta
            +str attn_type
            +Optional[int] num_attention_heads
            +Optional[int] num_key_value_heads
            +Optional[bool] use_qk_norm
            +Optional[bool] use_gated_attention
            +str ffn_type
            +Optional[dict] rope_scaling
            +Optional[str] pooling_type
            +Optional[bool] normalize_embeddings
        }

        class ConfigFactory {
            +Dict _entries
            +register(name) decorator
            +load(raw) BaseConfig
        }

        class InputConfig {
            +Optional[List[Dict]] sections
            +Optional[Dict[str, Dict]] sources
        }

        class ProcessingConfig {
            +int max_seq_len
            +int min_chars
            +int max_chars
            +Optional[int] max_items
            +str packing_strategy
            +int max_packed_len
            +str truncation_mode
        }

        class OutputConfig {
            +Optional[str] domain_key
            +str storage_format
            +int max_tokens_per_shard
            +Dict[str, str] dtype
            +str position_ids_mode
        }

        class PipelineConfig {
            +int version
            +InputConfig input
            +dict mask
            +str mask_default
            +ProcessingConfig preprocessing
            +OutputConfig output
            +from_dict(d) Self
        }

        class TrainConfig {
            +Callable[[], nn.Module] model_fn
            +str strategy
            +Dataset dataset
            +Callable optimizer_fn
            +Callable scheduler_fn
            +Optional[str] optimizer_name
            +Dict[str, Any] optimizer_hyperparameters
            +int n_epoch
            +int batch_per_device
            +int grad_accum_steps
            +Optional[float] max_grad_norm
            +list gradient_checkpointing_modules
            +Optional[str] compile_mode
            +int start_epoch
            +int start_samples
            +str ckpt_dir
            +int ckpt_interval
            +List[str] metrics
            +Optional[LoRAConfig] lora
            +int random_seed
            +int num_workers
            +Optional[int] prefetch_factor
            +bool pin_memory
            +Optional[Callable] collate_fn
            +int nprocs
            +str backend
            +str master_addr
            +str master_port
            +str start_method
            +str device_type
            +Optional[Dataset] val_dataset
            +Optional[float] val_split
            +int val_step
            +float neftune_alpha
            +float moe_aux_loss_coef
            +str parallel_mode
            +int rollout_interval
            +float rollout_temperature
            +int rollout_top_k
            +float rollout_top_p
            +int rollout_max_tokens
            +Optional[Callable] reward_model_fn
            +dict executor_kwargs
            +dict extra_kwargs
        }

    }

    namespace dataset {
        class BaseDataset {
            +int window_size
            +int stride
            +Optional[Store] storage
            +load(load_path, storage_type)
            +__getitem__(index)
            +__len__()
        }

        class SEQDataset {
            +__getitem__(index) Dict
        }

        class SFTDataset {
            +__getitem__(index) Dict
        }

        class DPODataset {
            +__getitem__(index) Dict
        }

        class GRPODataset {
            +__getitem__(index) Dict
        }

        class Store {
            +Dict[str, List[Tensor]] _data
            +Dict[str, List[int]] _cum
            +Dict[str, List[int]] _offsets
            +int _length
            +int _num_records
            +keys (property)
            +load(path)
            +__len__()
            -_normalize(raw, offsets)
        }

        class Streamable {
            <<mixin>>
            +fetch(begin, end, keys)
            -_fetch_stream_key(key, begin, end) Tensor
        }

        class Recordable {
            <<mixin>>
            +num_records (property)
            +fetch_record(index, keys)
            -_fetch_record_key(key, index) Tensor
        }

        class MmapStore {
            +List _mmap_refs
            +load(path)
        }

        class JsonlStore {
            +JsonlSource _source
            +Callable _processor
            +load(path, transform, processor)
            +fetch_record(index, keys)
        }

        class JsonlSource {
            +Path path
            +load() List[dict]
        }

        class RDSampler {
            +int epoch
            +int iter
        }

        class StoreFactory {
            +Dict _entries
            +register(name) decorator
            +create(storage_type) Store
        }

        class DatasetFactory {
            +Dict _entries
            +register(name) decorator
            +create(train_type, window_size, stride) BaseDataset
            +load(train_type, load_path, window_size, stride, storage_type, tokenizer_path, max_len, store) BaseDataset
        }
    }

    namespace serialization {
        class Checkpoint {
            +dict state_dict
            +int epoch
            +int consumed_samples
            +dict extra
            +dict meta
            +dict config
            +save(save_dir)
            +load(save_dir, broadcast, verify_checksums) Checkpoint
            +load_any(save_dir, broadcast, verify_checksums) Optional[Checkpoint]
        }
    }

    namespace model {
        class ModelFactory {
            +Dict _entries
            +register(name) decorator
            +get_component_class(name) Type
        }

        class AutoModel {
            <<nn.Module>>
            +BaseModelConfig config
            +from_pretrained(path, disable_random_init, strict) nn.Module
            +save_pretrained(save_directory)
            +to(*args, **kwargs) Self
        }

        class AutoRegressiveLM {
            +AutoRegressiveLMConfig config
            +RotaryEmbedding rotary_embedding
            +Embedding embed_tokens
            +ModuleList layers
            +RMSNorm norm
            +Linear lm_head
            +forward(input_ids, input_mask, kv_cache, position_ids) Dict[str, Tensor]
            +load_state_dict(state_dict, strict, assign)
            +state_dict()
        }

        class EmbeddingEncoder {
            +EncoderConfig config
            +RotaryEmbedding rotary_embedding
            +Embedding embed_tokens
            +ModuleList layers
            +RMSNorm norm
            +str pooling_type
            +bool normalize_embeddings
            +forward(input_ids, input_mask, position_ids) Tensor
            +load_state_dict(state_dict)
        }

        class DecoderBlock {
            +nn.Module attention  # GQA or MLA via AttnFactory
            +RMSNorm input_norm
            +nn.Module mlp        # MLP or DeepSeekMoE via FFNFactory
            +RMSNorm post_attention_norm
            +forward(x, rotary_emb, attention_mask, kv_cache, is_causal) DecoderOutput
        }

        class DecoderOutput {
            <<TypedDict>>
            +Tensor hidden_states
            +Optional[Tensor] aux_loss
            +Optional[RouterStats] router_stats
        }

        class GQA {
            +int dim
            +int n_heads
            +int n_kv_heads
            +int head_dim
            +int n_rep
            +int layer_id
            +bool use_qk_norm
            +bool use_gated_attention
            +Linear q_proj, k_proj, v_proj, o_proj
            +Linear gate  # only if use_gated_attention
            +RMSNorm q_norm, k_norm  # only if use_qk_norm
            +forward(x, rotary_emb, attn_mask, kv_cache, is_causal) Tensor
        }

        class MLA {
            +int dim
            +int n_heads
            +int n_kv_heads
            +int head_dim
            +int kv_lora_rank
            +int qk_nope_head_dim
            +int qk_rope_head_dim
            +int n_rep
            +int layer_id
            +bool use_qk_norm
            +bool use_gated_attention
            +Linear q_proj, kv_a_proj, kv_b_proj
            +Linear o_proj
            +Linear gate  # only if use_gated_attention
            +RMSNorm kv_norm
            +RMSNorm q_norm, k_norm  # only if use_qk_norm
            +forward(x, rotary_emb, attn_mask, kv_cache, is_causal) Tensor
        }

        class MLP {
            +Linear up, gate, down
            +forward(x) FFNOutput
        }

        class FFNOutput {
            <<TypedDict>>
            +Tensor hidden_states
            +Optional[Tensor] aux_loss
            +Optional[RouterStats] router_stats
        }

        class DeepSeekMoE {
            +int dim
            +int n_routed_experts
            +int n_shared_experts
            +int n_activated_experts
            +str topk_method
            +Linear router
            +ModuleList shared_experts
            +ModuleList routed_experts
            +forward(x) FFNOutput
        }

        class AttnFactory {
            +create(attn_type, **kwargs) nn.Module
        }

        class FFNFactory {
            +create(ffn_type, dim, dim_ffn, **kwargs) nn.Module
        }

        class RMSNorm {
            +Parameter weight
            +float norm_eps
            +tuple normalized_shape
            +forward(x) Tensor
        }

        class Linear {
            +Parameter weight
            +Optional[Parameter] bias  # only if bias=True
            +forward(x) Tensor
        }

        class RotaryEmbedding {
            +int dim
            +int max_len
            +float base
            +Optional[Dict] rope_scaling
            +Tensor freqs_cis
            +forward(x, position_ids=None) Tensor
        }

        class Embedding {
            +Parameter weight
            +float neftune_noise_alpha
            +forward(x) Tensor
            +set_neftune_alpha(alpha)
        }

        class LoRAConfig {
            +int r
            +int alpha
            +tuple target_modules
        }

        class LoRALinear {
            +Linear weight
            +Parameter lora_A, lora_B
            +forward(x) Tensor
            +merge()
        }
    }

    namespace preprocessing {
        class SectionRenderer {
            +process_sections(item, sections, config, tokenizer) Tuple
            +process_list_field(item, sections, config, tokenizer) Tuple
        }

        class BaseMaskBuilder {
            <<abstract>>
            +build(item, config, tokenizer) Optional[dict]
        }

        class SingleOutputMaskBuilder {
            +SectionRenderer renderer
            +build(item, config, tokenizer) Optional[dict]
        }

        class MultiOutputMaskBuilder {
            +SectionRenderer renderer
            +build(item, config, tokenizer) Optional[dict]
        }

        class SectionedMaskBuilder {
            +build(item, config, tokenizer) Optional[dict]
        }

        class PackingStrategy {
            <<abstract>>
            +apply(keys, max_packed_len, truncation_mode) Dict
        }

        class PackingStrategyFactory {
            +create(name, *args, **kwargs) PackingStrategy
        }

        class SimplePacking {
            +apply(keys, max_packed_len, truncation_mode) Dict
        }

        class BFDPacking {
            +apply(keys, max_packed_len, truncation_mode) Dict
        }

        class BFDSplitPacking {
            +apply(keys, max_packed_len, truncation_mode) Dict
        }

        class PositionIdStrategy {
            <<abstract>>
            +generate(sequences) List[int]
        }

        class PositionIdStrategyFactory {
            +create(name, *args, **kwargs) PositionIdStrategy
        }

        class NoPositionId {
            +generate(sequences) List[int]
        }

        class DocResetPositionId {
            +generate(sequences) List[int]
        }

        class ContinuousPositionId {
            +generate(sequences) List[int]
        }

        class StoreWriter {
            <<abstract>>
            +save(output_dir, domain, shard_idx, tensors)
        }

        class StoreWriterFactory {
            +create(name, *args, **kwargs) StoreWriter
        }

        class BinWriter {
            +save(output_dir, domain, shard_idx, tensors)
        }

        class Pipeline {
            +PipelineConfig config
            +List[str] paths
            +str output_dir
            +str tokenizer_path
            +AutoTokenizer tokenizer
            +BaseMaskBuilder mask_builder
            +PackingStrategy _packer
            +PositionIdStrategy _position_id
            +StoreWriter _writer
            +transform(item) Optional[dict]
            +run()
            +_flush(domains, shard_idx)
            +_inject_doc_reset_position_ids(keys, mode, seqs) Dict
            +_inject_continuous_position_ids(tensors, mode, seqs) Dict
            +_to_tensors(keys) Dict
        }

        class TokenizeTransform {
            +PipelineConfig config
            +AutoTokenizer tokenizer
            +BaseMaskBuilder mask_builder
            +PositionIdStrategy position_strategy
            +from_config_file(path) TokenizeTransform
            +apply(records) Dict[str, list]
        }
    }

    namespace tokenize {
        class AutoTokenizer {
            +vocab_size int
            +encode(tokens, out_ids, is_pretokenized, add_special_tokens) List
            +decode(tokens, skip_special_tokens) str
            +__getattr__(name) Any (bos_id, eos_id, pad_id, stop_ids)
            +apply_chat_template(messages, system_prompt, tokenize, add_generation_prompt) Union[str, List[int]]
            +set_chat_template(template)
            +load(path)
            +from_pretrained(path) AutoTokenizer
            +save_pretrained(save_path)
        }

        class ChatTemplate {
            +str template_str
            +render(messages, system_prompt, **extra_variables) str
            +from_string(template) ChatTemplate
        }
    }

    namespace factory {
        class BaseFactory {
            +Dict _entries
            +register(name) decorator
            +create(name, *args, **kwargs) T
            +get_component_class(name) Type
            +list_registered() list
            +is_registered(name) bool
        }

        class MaskBuilderFactory {
            +Dict _entries
            +register(name) decorator
            +create(name, *args, **kwargs) BaseMaskBuilder
        }
    }

    namespace trainer {
        class Trainer {
            +TrainConfig train_config
            +List[TrainCallback] callbacks
            +train(param_path=None, resume=False)
            -_get_default_callbacks() List[TrainCallback]
        }

        class TrainContext {
            +nn.Module model
            +BaseStrategy strategy
            +DataLoader dataloader
            +OptimizerProtocol optimizer
            +SchedulerProtocol scheduler
            +Checkpoint checkpoint
            +TrainConfig config
            +dict model_config
            +BaseExecutor executor
            +int epoch
            +int consumed_samples
            +float loss
            +Dict[str, float] metrics
            +Optional[float] grad_norm
            +GradSNRTracker grad_snr_tracker
            +DataLoader val_dataloader
            +Optional[float] val_loss
            +int world_size
            +int rank
            +dict kwargs
            +stop_requested (property) bool
            +optimizer_step (property) int
            +request_stop()
        }

        class TrainContextBuilder {
            +TrainConfig config
            +with_param_path(param_path, resume) TrainContextBuilder
            +build() TrainContext
        }

        class BaseStrategy {
            +Callable model
            +Optional[BaseExecutor] executor
            +float moe_aux_loss_coef
            +dict extra_kwargs
            +str device
            +__call__(batch) LossOutput
            +compute_loss(batch) Tensor
            +compute_loss_output(batch) LossOutput
            +supports_online() bool
            +set_rollout_runner(runner)
            +prepare_from_rollout(result) Dict
            +on_optimizer_step()
        }

        class LossOutput {
            <<TypedDict>>
            +Tensor loss
            +Dict[str, float] metrics
        }

        class StrategyFactory {
            +Dict _entries
            +register(name) decorator
            +create(train_type, model, device, **kwargs) BaseStrategy
        }

        class SEQStrategy {
            +float label_smoothing
            +compute_loss(batch) Tensor
        }

        class SFTStrategy {
            +float label_smoothing
            +compute_loss(batch) Tensor
        }

        class DPOStrategy {
            +nn.Module ref_model
            +float beta
            +str reduction
            +compute_loss(batch) Tensor
        }

        class GRPOStrategy {
            +nn.Module old_model
            +nn.Module ref_model
            +float clip_eps
            +float kl_coef
            +int group_size
            +compute_loss(batch) Tensor
            +sync_old_model()
        }

        class RawRollout {
            +Tensor prompts
            +Tensor prompt_mask
            +Tensor responses
            +Tensor response_mask
            +Tensor logprobs_old
            +int policy_version
            +List[str] prompt_texts
            +List[List[str]] response_texts
        }

        class RolloutResult {
            +Tensor rewards
        }

        class BaseRewardModel {
            <<abstract>>
            +score(List[str] prompts, List[List[str]] responses) Tensor
        }

        class RolloutGenerator {
            +InferenceScheduler scheduler
            +int max_tokens
            +int group_size
            +float temperature
            +int top_k
            +float top_p
            +float frequency_penalty
            +int rep_window
            +int policy_version
            +update_weights(policy_version) int
            +generate(batch) RawRollout
        }

        class RolloutRunner {
            +int policy_version
            +update_weights(policy_version) int
            +step()
            +clear_cache()
            +__call__(batch) Tuple[RolloutResult, bool]
        }

        class BaseScheduler {
            +get_lr() List[float]
            +step()
            +state_dict() dict
            +load_state_dict(d)
        }

        class SchedulerFactory {
            +Dict _entries
            +register(name) decorator
            +create(name, *args, **kwargs) BaseScheduler
        }

        class CosineScheduler {
            +int warmup_steps
            +int lr_decay_steps
            +int total_steps
            +float min_rate
        }

        class SGDRScheduler {
            +int warmup_steps
            +int cycle_length
            +float min_rate
            +int t_mult
        }

        class WSDScheduler {
            +int warmup_steps
            +int stable_steps
            +int decay_steps
            +float min_rate
        }

        class TrainCallback {
            <<protocol>>
            +on_train_begin(context)
            +on_train_end(context)
            +on_epoch_begin(context)
            +on_epoch_end(context)
            +on_batch_begin(context)
            +on_batch_end(context)
            +before_optimizer_step(context)
            +after_optimizer_step(context)
            +on_error(context)
        }

        class GradientClippingCallback {
            +Optional[float] max_grad_norm
            +before_optimizer_step(context)
        }

        class GradientCheckpointingCallback {
            +Optional[List[type]] modules
            +on_train_begin(context)
            +on_train_end(context)
        }

        class CheckpointCallback {
            +str save_dir
            +int interval
            +bool weight_only
            +Callable save_extra_fn
            -_save_checkpoint(context)
            +after_optimizer_step(context)
            +on_train_end(context)
            +on_error(context)
            +save_extra(context) dict
        }

        class ProgressBarCallback {
            +int num_epoch
            +int log_interval
            +IO file
            +tqdm progress_bar
            +on_epoch_begin(context)
            +before_optimizer_step(context)
            +on_epoch_end(context)
        }

        class MetricCallback {
            +Path ckpt_dir
            +int save_interval
            +List[str] metrics
            +int val_step
            +before_optimizer_step(context)
            +on_epoch_end(context)
            +on_train_end(context)
            +on_error(context)
            -_run_validation(context)
        }

        class CallbackFactory {
            +Dict _entries
            +register(name) decorator
            +create(name, **kwargs) TrainCallback
        }

    }

    namespace inference {
        class InferenceEngine {
            +nn.Module model
            +AutoTokenizer tokenizer
            +InferenceScheduler scheduler
            +generate(prompt, stream, max_tokens, temperature, top_p, top_k, frequency_penalty, rep_window) Union[Generator, str, List[str]]
            +generate_async(prompt, max_tokens, temperature, top_p, top_k, frequency_penalty, rep_window) AsyncGenerator
            +get_stats() Dict
            +shutdown()
        }

        class Executor {
            +AutoModel model
            +PagePool kv_cache
            +TaskCacheManager task_cache
            +InferenceWorkspace _workspace
            +Optional[str] device
            +Optional[torch.dtype] dtype
            +execute_prefill(tasks, start_pos=0)
            +execute_decode(tasks, return_logprobs=False) Union[List[int], List[Tuple[int, float]]]
        }

        class InferenceWorkspace {
            +int max_batch_size
            +int max_seq_len
            +torch.device device
            +torch.dtype dtype
            +Tensor arange
            +Tensor input_mask
            +Tensor input_ids
            +Tensor req_pool_indices
            +Tensor seq_lens
            +Tensor kv_indptr
            +Tensor qo_indptr
            +Tensor inc
            +Tensor out_cache_loc
            +fill_input_ids(ids) Tensor
            +decode_mask(position_ids, total_len) Tensor
        }

        class InferenceScheduler {
            +PagePool _cache
            +TaskCacheManager _task_cache
            +Executor _executor
            +TaskManager _task_mgr
            +Event _stop_event
            +Thread _loop_thread
            +int max_seq_len
            +str device
            +torch.dtype dtype
            +int policy_version
            +add_task(prompt, **kwargs) str
            +remove_task(task_id)
            +start()
            +stop()
            +get_stats() Dict
            +update_weights(policy_version) int
            +run_batch(prompt_ids_list, max_tokens, temperature, top_p, top_k, frequency_penalty, rep_window, return_logprobs) Union[List[List[int]], List[Tuple[List[int], List[float]]]]
        }

        class Allocator {
            +int _free_mask
            +List[int] _refs
            +OrderedDict _lru
            +alloc() int
            +free(idx, keep_cached)
            +inc_ref(idx)
            +touch(idx)
            +ref_count(idx) int
            +clear_cached() int
        }

        class RadixNode {
            +RadixNode parent
            +Dict children
            +Optional[int] page_idx
            +Tuple tokens
            +int lock_ref
        }

        class RadixCache {
            +int _page_size
            +evict(page_idx)
            +has_page(idx) bool
            +lookup(token_ids) List[int]
            +record(page_idx, token_ids, logical_page_idx)
            +release(pages)
        }

        class AllocationStrategy {
            <<abstract>>
            +alloc(state, prompt_ids) bool
            +free(state)
            +extend(state, pos) bool
            +write_indices(state, prompt_ids)
            +record_hashes(state, prompt_ids, start_logical_page)
            +invalidate_cache() int
        }

        class ContiguousStrategy {
            +write_indices(state, prompt_ids)
        }

        class PagedStrategy {
            -Allocator _alloc
            -RadixCache _prefix
        }

        class KVStorage {
            +int size
            +Tensor k_buffer
            +Tensor v_buffer
            +get_key_buffer(layer_id) Tensor
            +get_value_buffer(layer_id) Tensor
            +set_kv_buffer(layer_id, loc, k, v)
        }

        class ReqToTokenPool {
            +int size
            +int max_context_len
            +Tensor req_to_token
            +alloc(num_reqs) List[int]
            +free(req_indices)
            +write(indices, values)
        }

        class KVCache {
            +Tensor k_buffer
            +Tensor v_buffer
            +Tensor req_to_token
            +Tensor req_pool_indices
            +Tensor seq_lens
            +Tensor out_cache_loc
            +int max_len
            +Optional[Tensor] kv_indptr
            +Optional[Tensor] qo_indptr
            +Optional[Tensor] decode_o_part
            +Optional[Tensor] decode_ml_part
            +Optional[Tensor] decode_out
        }

        class PagePool {
            +int page_size
            +bool contiguous
            -KVStorage _storage
            -ReqToTokenPool _req_pool
            -AllocationStrategy _strategy
            +strategy AllocationStrategy
            +req_pool ReqToTokenPool
            +bind_tasks(req_indices, seq_lens, workspace, device, start_pos, incremental) KVCache
        }

        class TaskCacheManager {
            -PagePool _pool
            -Dict _states
            +task_alloc(task_id, prompt_ids) bool
            +task_free(task_id)
            +task_extend(task_id, pos) bool
            +task_cached(task_id) int
            +task_record_hashes(task_id, prompt_ids, start_logical_page)
            +invalidate_cache() int
            +bind(task_ids, workspace) KVCache
        }

    class Task {
        +str task_id
        +List prompt_ids
        +Optional[int] max_tokens
        +float temperature
        +float top_p
        +int top_k
        +float frequency_penalty
        +int rep_window
        +TaskStatus status
        +List output_ids
        +int input_tokens
        +int output_tokens
        +float arrival_time
        +Optional[float] finish_time
        +int next_pos
        +is_finished(stop_ids) bool
    }

        class TaskStatus {
            <<enumeration>>
            PENDING
            RUNNING
            FINISHED
            ABORTED
        }

        class TaskManager {
            +AutoTokenizer tokenizer
            +int max_batch_size
            +int max_seq_len
            +Deque waiting_queue
            +List active_tasks
            +add_task(prompt, max_tokens, temperature, top_p, top_k, stream_callback) str
            +remove_task(task_id) List[Task]
            +remove_finished_tasks(stop_ids) List[Task]
            +pull_candidates(n) List[Task]
            +activate(task)
            +return_to_waiting(tasks)
            +get_active_tasks() List[Task]
            +has_work() bool
            +wait_for_tasks(timeout)
            +get_waiting_tasks() List[Task]
            +clear_queues()
            +wake()
            +get_stats() Dict
        }

        class BaseSamplingStrategy {
            <<abstract>>
            +apply(logits, filter_value, input_ids, input_mask) Tensor
        }

        class TemperatureStrategy {
            +float temperature
            +apply(logits, filter_value, input_ids, input_mask) Tensor
        }

        class TopKStrategy {
            +int top_k
            +apply(logits, filter_value, input_ids, input_mask) Tensor
        }

        class TopPStrategy {
            +float top_p
            +apply(logits, filter_value, input_ids, input_mask) Tensor
        }

        class FrequencyPenaltyStrategy {
            +float penalty
            +apply(logits, filter_value, input_ids, input_mask) Tensor
        }

        class SamplingPipeline {
            +List[BaseSamplingStrategy] strategies
            +apply(logits, filter_value, input_ids, input_mask) Tensor
            +sample(logits, filter_value, input_ids, input_mask, return_logprobs) Union[Tensor, Tuple[Tensor, Tensor]]
        }

        class StreamDecoder {
            +push(token_id) str
        }

        class GenerateResult {
            +List[Tuple[int, str]] tokens
            +List[str] results
            +List[bool] _done
            +append(token, idx)
            +get_results() List[str]
            +pop_all() List[Tuple[int, str]]
            +wait(timeout) bool
            +wait_completion(timeout)
        }

        class ChatMessage {
            +str role
            +Optional[str] content
            +Optional[List[Dict]] tool_calls
            +Optional[str] tool_call_id
        }

        class FunctionDef {
            +str name
            +Optional[str] description
            +Optional[Dict] parameters
        }

        class ToolDef {
            +str type
            +FunctionDef function
        }

        class ChatCompletionRequest {
            +str model
            +List[ChatMessage] messages
            +Optional[float] temperature
            +Optional[float] top_p
            +Optional[int] top_k
            +Optional[int] max_tokens
            +Optional[bool] stream
            +Optional[Union[str, List[str]]] stop
            +Optional[int] n
            +Optional[float] presence_penalty
            +Optional[float] frequency_penalty
            +Optional[Dict[int, float]] logit_bias
            +Optional[str] user
            +Optional[List[ToolDef]] tools
            +Optional[Union[str, Dict]] tool_choice
        }

        class AnthropicMessage {
            +str role
            +Union[str, List[Dict]] content
        }

        class MessagesRequest {
            +str model
            +List[AnthropicMessage] messages
            +Optional[str] system
            +Optional[float] temperature
            +Optional[float] top_p
            +Optional[int] top_k
            +int max_tokens
            +Optional[bool] stream
            +Optional[List[str]] stop_sequences
        }

        class ResponseBuilder {
            <<abstract>>
            +prepare(request, engine) Tuple[str, GenContext, List[str]]
            +format_stream_start(ctx) List[str]
            +format_chunk(token, **kwargs) List[str]
            +format_stream_end(ctx, stop) List[str]
            +format_response(ctx, content, stop) Dict
        }

        class OpenAIResponseBuilder {
            +prepare(request, engine) Tuple
            +format_stream_start(ctx) List[str]
            +format_chunk(token, **kwargs) List[str]
            +format_stream_end(ctx, stop) List[str]
            +format_response(ctx, content, stop) Dict
        }

        class AnthropicResponseBuilder {
            +prepare(request, engine) Tuple
            +format_stream_start(ctx) List[str]
            +format_chunk(token, **kwargs) List[str]
            +format_stream_end(ctx, stop) List[str]
            +format_response(ctx, content, stop) Dict
        }

        class ProtocolHandler {
            +request
            +engine
            +builder: ResponseBuilder
            +async handle() Union[StreamingResponse, Dict]
            -_handle_stream(agen, ctx, stop_sequences) StreamingResponse
            -async _handle_non_stream(agen, ctx, stop_sequences) Dict
        }

        class StopChecker {
            +__init__(sequences)
            +check(text) Optional[str]
        }

        class GenContext {
            +str resp_id
            +int created
            +str model
            +int prompt_tokens
            +int completion_tokens
        }

        class StopInfo {
            +Optional[str] matched
            +str body
            +str yielded
        }

        class BaseToolParser {
            <<abstract>>
            +feed(body, current_token_ids, delta_token_ids) List[Dict]
            +parse_complete(body) Optional[Dict]
            +has_tool_calls (property) bool
        }

        class ToolParserFactory {
            +create(name, *args, **kwargs) BaseToolParser
        }

        class SimpleJsonToolParser {
            +feed(body, current_token_ids, delta_token_ids) List[Dict]
            +parse_complete(body) Optional[Dict]
        }
    }

    namespace protocols {
        class OptimizerProtocol {
            <<protocol>>
            +step(closure)
            +zero_grad()
            +state_dict() dict
            +load_state_dict(d)
        }

        class SchedulerProtocol {
            <<protocol>>
            +step()
            +state_dict() dict
            +load_state_dict(d)
            +get_last_lr()
        }
    }

    namespace parallel {
        class LaunchStrategy {
            <<abstract>>
            +launch(func, **kwargs)
        }

        class TorchrunStrategy {
            +launch(func, **kwargs)
        }

        class LocalStrategy {
            +launch(func, **kwargs)
        }

        class GradientState {
            +int num_steps
            +sync_gradients (property) bool
        }

        class AccumOptimizer {
            +Optimizer optimizer
            +GradientState gradient_state
            +param_groups (property)
            +step(closure)
            +zero_grad()
            +state_dict() dict
            +load_state_dict(d)
        }

        class AccumScheduler {
            +LRScheduler scheduler
            +GradientState gradient_state
            +step()
            +state_dict() dict
            +load_state_dict(d)
            +get_last_lr()
        }

        class RolloutCapabilities {
            +bool supports_in_process
            +Optional[str] reason
        }

        class BaseExecutor {
            +GradientState gradient_state
            +prepare(model_fn, optimizer_fn, scheduler_fn, before_wrap, after_wrap) tuple
            +rollout_capabilities() RolloutCapabilities
            +model_for_inference(model) nn.Module
            +accumulate(model) context manager
            +backward(loss)
            +unwrap_model(model) dict
            +checkpoint_context(model) context manager
            +clip_grad_norm(model, max_norm) float
            +use_distributed (property) bool
            +sync_gradients (property) bool
            +grad_accum_steps (property) int
        }

        class NoneExecutor {
        }

        class DDPExecutor {
            -_prepare_model(model) nn.Module
            -_no_sync(model) context manager
            +model_for_inference(model) nn.Module
            +unwrap_model(model) dict
        }

        class FSDPExecutor {
            -_prepare_model(model) nn.Module
            -_no_sync(model) context manager
            +rollout_capabilities() RolloutCapabilities
            +unwrap_model(model) Optional[dict]
            +clip_grad_norm(model, max_norm) float
        }

        class ExecutorFactory {
            +Dict _entries
            +register(name) decorator
            +create(parallel_mode, **kwargs) BaseExecutor
        }

    }

    %% Relationships — UML notation: <|-- generalization, *-- composition, o-- aggregation, --> association, ..> dependency

    %% --- Generalization (inheritance) ---
    BaseStrategy <|-- SEQStrategy
    BaseStrategy <|-- SFTStrategy
    BaseStrategy <|-- DPOStrategy
    BaseStrategy <|-- GRPOStrategy
    BaseScheduler <|-- CosineScheduler
    BaseScheduler <|-- SGDRScheduler
    BaseScheduler <|-- WSDScheduler
    TrainCallback <|-- GradientClippingCallback
    TrainCallback <|-- GradientCheckpointingCallback
    TrainCallback <|-- CheckpointCallback
    TrainCallback <|-- ProgressBarCallback
    TrainCallback <|-- MetricCallback
    BaseDataset <|-- SEQDataset
    BaseDataset <|-- SFTDataset
    BaseDataset <|-- DPODataset
    BaseDataset <|-- GRPODataset
    Store <|-- MmapStore
    Store <|-- JsonlStore
    MmapStore --|> Streamable
    MmapStore --|> Recordable
    JsonlStore --|> Streamable
    JsonlStore --|> Recordable
    BaseSamplingStrategy <|-- TemperatureStrategy
    BaseSamplingStrategy <|-- TopKStrategy
    BaseSamplingStrategy <|-- TopPStrategy
    BaseSamplingStrategy <|-- FrequencyPenaltyStrategy
    AutoModel <|-- AutoRegressiveLM
    AutoModel <|-- EmbeddingEncoder
    BaseConfig <|-- BaseModelConfig
    BaseConfig <|-- TrainConfig
    BaseConfig <|-- InputConfig
    BaseConfig <|-- ProcessingConfig
    BaseConfig <|-- OutputConfig
    BaseConfig <|-- PipelineConfig
    BaseModelConfig <|-- AutoRegressiveLMConfig
    BaseModelConfig <|-- EncoderConfig
    BaseFactory <|-- ModelFactory
    BaseFactory <|-- AttnFactory
    BaseFactory <|-- FFNFactory
    BaseFactory <|-- DatasetFactory
    BaseFactory <|-- StrategyFactory
    BaseFactory <|-- SchedulerFactory
    BaseFactory <|-- CallbackFactory
    BaseFactory <|-- StoreFactory
    BaseFactory <|-- ExecutorFactory
    BaseFactory <|-- ConfigFactory
    BaseFactory <|-- MaskBuilderFactory
    BaseFactory <|-- PackingStrategyFactory
    BaseFactory <|-- PositionIdStrategyFactory
    BaseFactory <|-- StoreWriterFactory
    BaseFactory <|-- ToolParserFactory
    BaseExecutor <|-- NoneExecutor
    BaseExecutor <|-- DDPExecutor
    BaseExecutor <|-- FSDPExecutor
    ResponseBuilder <|-- OpenAIResponseBuilder
    ResponseBuilder <|-- AnthropicResponseBuilder
    BaseToolParser <|-- SimpleJsonToolParser
    BaseMaskBuilder <|-- SectionedMaskBuilder
    BaseMaskBuilder <|-- SingleOutputMaskBuilder
    BaseMaskBuilder <|-- MultiOutputMaskBuilder
    PackingStrategy <|-- SimplePacking
    PackingStrategy <|-- BFDPacking
    BFDPacking <|-- BFDSplitPacking
    PositionIdStrategy <|-- NoPositionId
    PositionIdStrategy <|-- DocResetPositionId
    PositionIdStrategy <|-- ContinuousPositionId
    StoreWriter <|-- BinWriter
    AllocationStrategy <|-- ContiguousStrategy
    AllocationStrategy <|-- PagedStrategy
    RawRollout <|-- RolloutResult
    LaunchStrategy <|-- TorchrunStrategy
    LaunchStrategy <|-- LocalStrategy
    %% --- Composition (strong ownership, part destroyed with whole) ---
    PagePool *-- KVStorage
    PagePool *-- ReqToTokenPool
    PagePool *-- AllocationStrategy
    PagedStrategy *-- Allocator
    PagedStrategy *-- RadixCache
    TaskCacheManager o-- PagePool
    RadixCache *-- RadixNode
    InferenceEngine *-- InferenceScheduler
    InferenceScheduler *-- PagePool
    InferenceScheduler *-- TaskCacheManager
    InferenceScheduler *-- Executor
    Executor *-- InferenceWorkspace
    InferenceScheduler *-- TaskManager
    AutoRegressiveLM *-- DecoderBlock
    AutoRegressiveLM *-- RotaryEmbedding
    AutoRegressiveLM *-- Embedding
    EmbeddingEncoder *-- DecoderBlock
    EmbeddingEncoder *-- RotaryEmbedding
    EmbeddingEncoder *-- Embedding
    DecoderBlock *-- RMSNorm
    ChatCompletionRequest *-- ChatMessage
    ChatCompletionRequest *-- ToolDef
    ToolDef *-- FunctionDef
    MessagesRequest *-- AnthropicMessage
    BaseExecutor *-- GradientState
    AccumOptimizer o-- GradientState
    AccumScheduler o-- GradientState

    %% --- Aggregation (weak ownership) ---
    AutoModel o-- BaseModelConfig
    AutoTokenizer o-- ChatTemplate
    Trainer o-- TrainCallback
    TrainContext o-- BaseStrategy
    TrainContext o-- BaseScheduler
    TrainContext o-- Checkpoint
    TrainContext o-- BaseExecutor
    SamplingPipeline o-- BaseSamplingStrategy
    BaseDataset o-- Store
    Pipeline o-- PipelineConfig
    Pipeline o-- BaseMaskBuilder
    Pipeline o-- AutoTokenizer
    Pipeline o-- PackingStrategy
    Pipeline o-- PositionIdStrategy
    Pipeline o-- StoreWriter
    TokenizeTransform o-- AutoTokenizer
    TokenizeTransform o-- BaseMaskBuilder

    %% --- Dependency (uses temporarily) ---
    TrainConfig ..> BaseStrategy : selects
    PipelineConfig ..> MaskBuilderFactory : selects
    MaskBuilderFactory ..> BaseMaskBuilder : creates
    PackingStrategyFactory ..> PackingStrategy : creates
    PositionIdStrategyFactory ..> PositionIdStrategy : creates
    StoreWriterFactory ..> StoreWriter : creates
    StrategyFactory ..> BaseStrategy : creates
    SchedulerFactory ..> BaseScheduler : creates
    DatasetFactory ..> BaseDataset : creates
    CallbackFactory ..> TrainCallback : creates
    AttnFactory ..> GQA : creates
    AttnFactory ..> MLA : creates
    FFNFactory ..> MLP : creates
    FFNFactory ..> DeepSeekMoE : creates
    DecoderBlock ..> AttnFactory : uses
    DecoderBlock ..> FFNFactory : uses
    StoreFactory ..> MmapStore : creates
    StoreFactory ..> JsonlStore : creates
    ConfigFactory ..> AutoRegressiveLMConfig : creates
    ConfigFactory ..> EncoderConfig : creates
    ModelFactory ..> AutoRegressiveLM : creates
    ModelFactory ..> EmbeddingEncoder : creates
    ExecutorFactory ..> NoneExecutor : creates
    ExecutorFactory ..> DDPExecutor : creates
    ExecutorFactory ..> FSDPExecutor : creates
    ToolParserFactory ..> BaseToolParser : creates
    TrainContextBuilder ..> ExecutorFactory : creates
    Trainer ..> TrainContextBuilder : uses
    TrainContextBuilder ..> TrainContext : creates
    TrainContextBuilder ..> StrategyFactory : uses
    TrainContextBuilder ..> RDSampler : creates
    Checkpoint ..> Checkpoint : serializes
    CheckpointCallback ..> Checkpoint : creates
    PagePool ..> KVCache : binds
    PagePool ..> InferenceWorkspace : fills
    InferenceEngine ..> GenerateResult : uses
    InferenceEngine ..> GenerateResult : creates
    OpenAIResponseBuilder ..> ChatCompletionRequest : receives
    AnthropicResponseBuilder ..> MessagesRequest : receives
    ProtocolHandler ..> StopChecker : creates
    ProtocolHandler ..> GenContext : creates
    RolloutGenerator ..> InferenceScheduler : uses
    RolloutRunner ..> RolloutGenerator : uses
    RolloutRunner ..> BaseRewardModel : uses

    %% --- Association (general usage) ---
    Trainer --> TrainConfig
    DPOStrategy --> AutoModel
    GRPOStrategy --> AutoModel : policy/old/ref
    InferenceScheduler --> Task
    InferenceScheduler --> TaskStatus
    Task --> TaskStatus
    InferenceEngine --> AutoModel
    Executor --> AutoModel
    Executor --> TaskCacheManager
    TaskManager --> AutoTokenizer

```


## Module Overview

| Module | Components | Description |
|--------|------------|-------------|
| **astrai.config** | BaseConfig, BaseModelConfig, AutoRegressiveLMConfig, EncoderConfig, ConfigFactory, TrainConfig, PipelineConfig, InputConfig, ProcessingConfig, OutputConfig | Configuration management (to_dict/from_dict, to_file/from_file) |
| **astrai.preprocessing** | SectionRenderer, BaseMaskBuilder, MaskBuilderFactory, SectionedMaskBuilder, SingleOutputMaskBuilder, MultiOutputMaskBuilder, Pipeline, TokenizeTransform, PackingStrategy, PackingStrategyFactory, SimplePacking, BFDPacking, BFDSplitPacking, PositionIdStrategy, PositionIdStrategyFactory, NoPositionId, DocResetPositionId, ContinuousPositionId, StoreWriter, StoreWriterFactory, BinWriter | Declarative JSON-driven data preprocessing |
| **astrai.dataset** | BaseDataset, SEQDataset, SFTDataset, DPODataset, GRPODataset, Store, Streamable, Recordable, MmapStore, JsonlSource, JsonlStore, StoreFactory, RDSampler, DatasetFactory | Dataset loading and management |
| **astrai.serialization** | Checkpoint | Model serialization |
| **astrai.model** | ModelFactory, AutoModel, AutoRegressiveLM, EmbeddingEncoder, DecoderBlock, GQA, MLA, MLP, DeepSeekMoE, AttnFactory, FFNFactory, RMSNorm, Linear, LoRAConfig, LoRALinear, RotaryEmbedding, Embedding | Neural network model |
| **astrai.tokenize** | AutoTokenizer, ChatTemplate | Tokenizer and chat template |
| **astrai.trainer** | Trainer, TrainContext, TrainContextBuilder, BaseStrategy–GRPOStrategy, StrategyFactory, BaseScheduler–WSDScheduler, SchedulerFactory, TrainCallback(Protocol)–MetricCallback, CallbackFactory, RawRollout, RolloutResult, BaseRewardModel, RolloutGenerator, RolloutRunner | Training workflow |
| **astrai.inference** | InferenceEngine, InferenceScheduler, Executor, InferenceWorkspace, PagePool, TaskCacheManager, KVStorage, ReqToTokenPool, KVCache, Allocator, RadixCache, AllocationStrategy, ContiguousStrategy, PagedStrategy, Task, TaskManager, TaskStatus, StreamDecoder, GenerateResult, BaseSamplingStrategy–SamplingPipeline, FrequencyPenaltyStrategy, ProtocolHandler, ResponseBuilder, OpenAIResponseBuilder, AnthropicResponseBuilder, StopChecker, GenContext, StopInfo, ChatMessage, FunctionDef, ToolDef, ChatCompletionRequest, AnthropicMessage, MessagesRequest, BaseToolParser, ToolParserFactory, SimpleJsonToolParser | Inference service |
| **astrai.extension** | `backend` policy package, `ops` kernel-wrapper package, `fp8.py` FP8 strategy layer, AttentionBackend, TorchNativeBackend, CudaBackend, FlashAttnBackend, attention, attn_backend, ATTN_BACKEND, apply_rotary_emb, is_available | Stable API over attention/rotary/FP8 execution policy and optional CUDA kernels |
| **astrai.optim** | OptimizerFactory, MuonAdamW, NoraNadamW, ManoAdamW, composite_step/composite_zero_grad/composite_state_dict, partition_optimizer_parameters | Built-in optimizers (`muon_adamw` / `nora_nadamw` / `mano_adamw`) with shared composite-optimizer helpers |
| **astrai.parallel** | spawn_parallel_fn, setup_parallel, get_rank/get_world_size/get_current_device, only_on_rank, LaunchStrategy, TorchrunStrategy, LocalStrategy, BaseExecutor, ExecutorFactory, NoneExecutor, DDPExecutor, FSDPExecutor, GradientState, AccumOptimizer, AccumScheduler | Distributed parallel & gradient accumulation |
| **astrai.factory** | BaseFactory | Component registration |
| **astrai.protocols** | OptimizerProtocol, SchedulerProtocol | Structural subtyping for optimizer/scheduler wrappers |

## Design Patterns

| Pattern | Classes | Purpose |
|---------|---------|---------|
| **Factory** | `ModelFactory`, `AttnFactory`, `FFNFactory`, `StrategyFactory`, `DatasetFactory`, `SchedulerFactory`, `CallbackFactory`, `StoreFactory`, `ConfigFactory`, `ExecutorFactory`, `MaskBuilderFactory`, `StoreWriterFactory`, `PackingStrategyFactory`, `PositionIdStrategyFactory`, `ToolParserFactory` | Decorator-based component creation |
| **Registry** | `BaseFactory` | Component registration |
| **Strategy** | `SEQStrategy`, `SFTStrategy`, `DPOStrategy`, `GRPOStrategy` | Training strategy switching |
| **Strategy (Sampling)** | `TemperatureStrategy`, `TopKStrategy`, `TopPStrategy`, `FrequencyPenaltyStrategy`, `SamplingPipeline` | Composable logit transformations |
| **Strategy (API)** | `ResponseBuilder`, `OpenAIResponseBuilder`, `AnthropicResponseBuilder` | HTTP API handler with format hooks |
| **Builder** | `TrainContextBuilder` | Chain-building training context |
| **Observer** | `TrainCallback`, callback implementations | Training process monitoring |
| **Context** | `TrainContext` | Unified training state bag |
| **Object Pool** | `Allocator`, `PagePool` | Page-based KV cache with LRU eviction |
| **Strategy (Attention)** | `AttentionBackend`, `CudaBackend`, `FlashAttnBackend`, `TorchNativeBackend` | Attention computation backend switching via context manager |
| **Auto-dispatch (Rotary)** | `apply_rotary_emb`, `backend/rotary.py`, `ops/rotary.py` | Rotary embedding CUDA kernel auto-dispatch with torch fallback |
| **Executor** | `BaseExecutor`, `NoneExecutor`, `DDPExecutor`, `FSDPExecutor` | Gradient accumulation & model distribution |
| **Storage** | `Store`, `MmapStore`, `JsonlStore` | Format-agnostic data access with multi-segment support |
| **Producer-Consumer** | `InferenceScheduler`, `Task`, queues | Continuous batching |
| **Model Registry** | `ModelFactory`, `AutoRegressiveLM`, `EmbeddingEncoder` | Model-type dynamic loading |
| **Optimizer Routing** | `OptimizerFactory`, `MuonAdamW`, `NoraNadamW`, `ManoAdamW` | Route parameter groups (matrices vs. embeddings/heads/norms) through different optimizers |

## Core Relationships

1. **Config → Training**: `TrainConfig` holds `model_fn`, `dataset`, `optimizer_fn`, `scheduler_fn`, `parallel_mode`, `executor_kwargs`
2. **Training Flow**: `Trainer` → `TrainContextBuilder` → `TrainContext`, uses `BaseStrategy` for loss, `BaseExecutor` for gradient accumulation + model distribution
3. **Strategy Selection**: `StrategyFactory` creates strategy by `train_type`
4. **Executor Selection**: `ExecutorFactory.create(cfg.parallel_mode, grad_accum_steps=cfg.grad_accum_steps, **cfg.executor_kwargs)` → `NoneExecutor` / `DDPExecutor` / `FSDPExecutor`
5. **Inference Flow**: `InferenceEngine` → `InferenceScheduler` → `AutoRegressiveLM`, backed by `PagePool` + `KVCache` + `SamplingPipeline`. `astrai.extension.backend` owns attention/rotary dispatch, fallback, and KV cache policy; it calls the stateless compiled-kernel wrappers in `astrai.extension.ops`. Attention uses cuda > flash > torch priority unless explicitly selected by `ASTR_BACKEND` or `attn_backend()`. Rotary embedding auto-dispatches to the CUDA op when supported, else torch complex multiply.
6. **Distributed**: `spawn_parallel_fn` + `setup_parallel` for multi-process DDP. Online rollout obtains an explicit inference-model view from the training executor: DDP exposes its replicated underlying module, while distributed FSDP and `torch.compile` fail before scheduler construction.
7. **Dataset Loading**: `DatasetFactory` creates datasets, `Store` (`MmapStore`/`JsonlStore`) loads data with explicit `_length` and multi-segment `_data`
8. **Checkpoint**: `Checkpoint` saves/loads safetensors + metadata; `CheckpointCallback` performs rank-0 training saves, with extra state saved as `{key}.pt`
9. **Scheduler**: `SchedulerFactory` creates `CosineScheduler`/`SGDRScheduler`/`WSDScheduler`
10. **AutoModel**: `from_pretrained()` loads `config.json` + `model.safetensors`, `_disable_random_init` replaces `nn.init.*` with no-ops
11. **Protocols**: `OptimizerProtocol` / `SchedulerProtocol` — structural subtyping for `AccumOptimizer` / `AccumScheduler` wrappers

> Document Update Time: 2026-08-29
