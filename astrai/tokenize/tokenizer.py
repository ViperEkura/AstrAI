"""
Tokenizer module with implementation and auto-loading support.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import DecodeStream

from astrai.tokenize.chat_template import ChatTemplate, ChatTemplateInputError

Message = Dict[str, Any]
"""Single chat message with ``role`` and ``content`` keys."""

Messages = List[Message]
"""Single conversation — a list of messages."""


class AutoTokenizer:
    """Base tokenizer class with automatic loading support"""

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        special_token_map: Optional[Dict[str, str]] = None,
        chat_template: Optional[str] = None,
    ):
        self._tokenizer: Optional[Tokenizer] = None
        self._chat_template: Optional[ChatTemplate] = None
        self._special_token_map = dict(special_token_map or {})
        self._tokenizer_config: Dict = {}
        if path:
            self.load(
                path, special_token_map=special_token_map, chat_template=chat_template
            )
        elif chat_template:
            self.set_chat_template(chat_template)

    def load(
        self,
        path: Union[str, Path],
        *,
        special_token_map: Optional[Dict[str, str]] = None,
        chat_template: Optional[str] = None,
    ):
        """Replace state after parsing succeeds; file fields override explicit defaults."""
        path = Path(path)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        configs = {}
        for name in ("special_tokens_map", "tokenizer_config"):
            file = path / f"{name}.json"
            configs[name] = (
                json.loads(file.read_text(encoding="utf-8")) if file.is_file() else {}
            )
        config = {**configs["special_tokens_map"], **configs["tokenizer_config"]}
        if "special_tokens" in config or "stop_token_ids" in config:
            raise ValueError("Legacy tokenizer fields require migration to HF format")
        for token_id, spec in sorted(
            config.get("added_tokens_decoder", {}).items(),
            key=lambda item: int(item[0]),
        ):
            token = AddedToken(**{k: v for k, v in spec.items() if k != "__type"})
            tokenizer.add_tokens([token])
            if tokenizer.token_to_id(token.content) != int(token_id):
                raise ValueError(f"Added token ID mismatch: {token.content!r}")
        special_tokens = dict(special_token_map or {})
        extra_tokens = config.get("extra_special_tokens", {})
        decoder = tokenizer.get_added_tokens_decoder()
        for name in (
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "sep_token",
            "cls_token",
            "mask_token",
            "additional_special_tokens",
            *extra_tokens,
        ):
            value = config.get(name, extra_tokens.get(name, special_tokens.get(name)))
            if value is None:
                special_tokens.pop(name, None)
                continue
            values = value if isinstance(value, list) else [value]
            tokens = []
            for item in values:
                if isinstance(item, dict):
                    token = AddedToken(
                        **{k: v for k, v in item.items() if k != "__type"}
                    )
                else:
                    token = decoder.get(
                        tokenizer.token_to_id(item), AddedToken(item, special=True)
                    )
                tokens.append(token)
            tokenizer.add_special_tokens(tokens)
            contents = [token.content for token in tokens]
            special_tokens[name] = contents if isinstance(value, list) else contents[0]
        template = config.get("chat_template", chat_template)
        template_file = path / "chat_template.jinja"
        if template_file.is_file():
            template = template_file.read_text(encoding="utf-8")
        template = (
            ChatTemplate.from_string(template, special_tokens=special_tokens)
            if template is not None
            else None
        )
        self._tokenizer = tokenizer
        self._special_token_map = special_tokens
        self._chat_template = template
        self._tokenizer_config = config

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "AutoTokenizer":
        """Load tokenizer from pretrained directory.

        Raises:
            FileNotFoundError: If tokenizer.json is missing.
        """
        path = Path(path)
        tokenizer_file = path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found: {tokenizer_file}. "
                "A valid tokenizer.json is required."
            )
        return cls(path)

    def save_pretrained(self, save_path: str):
        """Save HF tokenizer files; generation configuration belongs to the model."""
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer not initialized. Load or create a tokenizer first."
            )

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save tokenizer
        self._tokenizer.save(str(save_path / "tokenizer.json"))

        config = {**self._tokenizer_config, **self._special_token_map}
        config.setdefault("tokenizer_class", "PreTrainedTokenizerFast")
        config["added_tokens_decoder"] = {
            str(token_id): token.__getstate__()
            for token_id, token in self._tokenizer.get_added_tokens_decoder().items()
        }
        config.pop("chat_template", None)
        for name, value in (
            ("tokenizer_config.json", config),
            ("special_tokens_map.json", self._special_token_map),
        ):
            (save_path / name).write_text(
                json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        template_file = save_path / "chat_template.jinja"
        if self._chat_template is None:
            template_file.unlink(missing_ok=True)
        else:
            template_file.write_text(self._chat_template.template_str, encoding="utf-8")

    def encode(
        self,
        tokens: Union[str, List[str]],
        out_ids: bool = True,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> List:
        """Encode text to token IDs.

        Accepts both single strings and batches:

        - ``encode("hello")`` → ``[123, 456]``
        - ``encode(["hello", "world"])`` → ``[[123, 456], [789]]``

        Batches use the Rust backend fast ``encode_batch_fast`` path.
        This wrapper returns IDs or token strings, so character offsets are
        not needed.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer not initialized. Load or create a tokenizer first."
            )

        if isinstance(tokens, str):
            encoded = self._tokenizer.encode(
                tokens,
                is_pretokenized=is_pretokenized,
                add_special_tokens=add_special_tokens,
            )
            return encoded.ids if out_ids else encoded.tokens

        encoded_list = self._tokenizer.encode_batch_fast(
            tokens,
            is_pretokenized=is_pretokenized,
            add_special_tokens=add_special_tokens,
        )
        return [encoded.ids if out_ids else encoded.tokens for encoded in encoded_list]

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer not initialized. Load or create a tokenizer first."
            )

        return self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def __len__(self) -> int:
        if self._tokenizer is None:
            return 0
        return self._tokenizer.get_vocab_size()

    def __getattr__(self, key: str):
        """Expose named token strings and IDs; reject private probes during unpickle."""
        if key.startswith("_"):
            raise AttributeError(key)

        if key == "additional_special_tokens_ids":
            return [
                self._tokenizer.token_to_id(token)
                for token in self._special_token_map.get(
                    "additional_special_tokens", []
                )
            ]

        # Handle _id suffix (e.g., bos_token_id -> bos_token)
        if key.endswith("_id"):
            base_attr = key[:-3]  # Remove "_id"
            token_str = self._special_token_map.get(base_attr)
            if token_str is None:
                return None
            if self._tokenizer is None:
                raise RuntimeError("Tokenizer not loaded, cannot convert token to id.")
            return self._tokenizer.token_to_id(token_str)

        # Handle regular string attributes
        if key in self._special_token_map:
            return self._special_token_map.get(key)

        # Other attributes trigger default AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    @property
    def all_special_ids(self) -> List[int]:
        return list(
            dict.fromkeys(
                self._tokenizer.token_to_id(token)
                for value in self._special_token_map.values()
                for token in (value if isinstance(value, list) else [value])
            )
        )

    @property
    def vocab_size(self) -> int:
        return len(self)

    def set_chat_template(self, template: Union[str, ChatTemplate]):
        """Set a Jinja template string or an existing ChatTemplate."""
        if isinstance(template, str):
            self._chat_template = ChatTemplate.from_string(
                template, special_tokens=self._special_token_map
            )
        elif isinstance(template, ChatTemplate):
            self._chat_template = template
        else:
            raise ValueError("Invalid template type, must be str or ChatTemplate.")

    def apply_chat_template(
        self,
        messages: Union[Messages, List[Messages]],
        system_prompt: Optional[str] = None,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        return_assistant_tokens_mask: bool = False,
        enable_thinking: Optional[bool] = None,
        **kwargs,
    ) -> Union[str, List[int], List[str], List[List[int]], Dict[str, Any]]:
        """Apply the chat template and optionally tokenize.

        Accepts both single conversations and batches:

        - ``apply_chat_template([msg1, msg2])`` → ``"..."`` or ``[ids]``
        - ``apply_chat_template([[msg1, msg2], [msg3]])`` → ``["..", ".."]``
          or ``[[ids], [ids]]``

        Batches render each conversation list and tokenise all at once via
        :meth:`encode` (``List[str]`` → Rust parallel ``encode_batch``).

        Args:
            messages: Single conversation (``Messages``) or batch of
                conversations (``BatchMessages``).
            system_prompt: Optional system prompt prepended (single mode only).
            tokenize: Whether to return token IDs (True) or raw string (False).
            add_generation_prompt: Whether to add the generation prompt.
            return_assistant_tokens_mask: Return input_ids and assistant_masks
                from template generation blocks, using full-text token offsets.
            enable_thinking: Optional per-request thinking mode. Omit to use the template default.
            **kwargs: Additional template variables.

        Returns:
            Single mode: ``str`` or ``List[int]``.
            Batch mode: ``List[str]`` or ``List[List[int]]``.
        """
        if self._chat_template is None:
            raise RuntimeError(
                "Chat template not set. Use set_chat_template() to set a template first."
            )

        suffix = ""
        if enable_thinking is not None:
            if not isinstance(enable_thinking, bool):
                raise ChatTemplateInputError("enable_thinking must be a boolean")
            if "enable_thinking" in self._chat_template.template_variables:
                kwargs["enable_thinking"] = enable_thinking
            elif add_generation_prompt:
                if self._tokenizer is not None and all(
                    self._tokenizer.token_to_id(tag) is not None
                    for tag in ("<think>", "</think>")
                ):
                    suffix = (
                        "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
                    )
                elif enable_thinking:
                    raise ChatTemplateInputError(
                        "Thinking requires a supporting chat template or thinking tokens"
                    )

        is_batch = bool(messages) and isinstance(messages[0], list)
        if return_assistant_tokens_mask:
            if not tokenize:
                raise ValueError("Assistant token masks require tokenize=True")
            if self._tokenizer is None:
                raise RuntimeError("Tokenizer not initialized")

        if system_prompt and not is_batch:
            messages = [{"role": "system", "content": system_prompt}] + list(messages)
        conversations = messages if is_batch else [messages]
        kwargs["add_generation_prompt"] = add_generation_prompt
        rendered = []
        outputs = []
        for conversation in conversations:
            if return_assistant_tokens_mask:
                text, spans = self._chat_template.render_with_spans(
                    conversation, **kwargs
                )
            else:
                text = self._chat_template.render(conversation, **kwargs)
            if suffix:
                if not text.endswith("<|im_start|>assistant\n"):
                    raise ChatTemplateInputError(
                        "This generation prompt does not support switching thinking mode"
                    )
                text += suffix
            if not return_assistant_tokens_mask:
                rendered.append(text)
                continue

            merged_spans = []
            for start, end in spans:
                if merged_spans and start == merged_spans[-1][1]:
                    merged_spans[-1] = (merged_spans[-1][0], end)
                elif end > start:
                    merged_spans.append((start, end))
            encoded = self._tokenizer.encode(text, add_special_tokens=False)
            mask = []
            for start, end in encoded.offsets:
                overlaps = [
                    (a, b)
                    for a, b in merged_spans
                    if start < b and end > a and end > start
                ]
                if overlaps and not any(a <= start and end <= b for a, b in overlaps):
                    raise ValueError(
                        "A token crosses a generation boundary; use token-aligned delimiters"
                    )
                mask.append(int(bool(overlaps)))
            outputs.append({"input_ids": encoded.ids, "assistant_masks": mask})
        if return_assistant_tokens_mask:
            if is_batch:
                return {
                    key: [output[key] for output in outputs]
                    for key in ("input_ids", "assistant_masks")
                }
            return outputs[0]
        result = rendered if is_batch else rendered[0]
        return self.encode(result, add_special_tokens=False) if tokenize else result


class StreamDecoder:
    """Incremental decoder backed by the tokenizers library's DecodeStream.

    Delegates to the Rust-native streaming decoder which maintains an
    O(1) bounded token buffer internally (via prefix drain), avoiding
    the O(n²) cost of re-decoding the full history on each step.

    Multi-byte UTF-8 sequences split across token boundaries are
    buffered until complete; ``push`` returns "" while the trailing
    sequence is still incomplete.
    """

    __slots__ = ("_stream", "_tok")

    def __init__(self, tokenizer: AutoTokenizer):
        # Test doubles and lightweight tokenizers may lack the Rust handle;
        # ``push`` degrades to id-as-string so streams still terminate.
        self._tok = getattr(tokenizer, "_tokenizer", None)
        self._stream = (
            DecodeStream(skip_special_tokens=True) if self._tok is not None else None
        )

    def push(self, token_id: int) -> str:
        """Append a token ID and return newly completed text.

        Returns "" while a multi-byte character is still incomplete.
        """
        if self._stream is None:
            return str(token_id)
        chunk = self._stream.step(self._tok, token_id)
        return chunk or ""
