"""
Tokenizer module with implementation and auto-loading support.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from tokenizers import Tokenizer

from astrai.tokenize.chat_template import ChatTemplate


class AutoTokenizer:
    """Base tokenizer class with automatic loading support"""

    TOKENIZER_CLASSES = {}  # Registry for auto-loading

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        special_token_map: Optional[Dict[str, str]] = None,
        chat_template: Optional[str] = None,
    ):
        self._tokenizer: Tokenizer = None
        self._chat_template: Optional[ChatTemplate] = None
        self._special_token_map: Optional[Dict] = special_token_map or {}

        if chat_template:
            self.set_chat_template(chat_template)

        if path:
            self.load(path)

    def load(self, path: Union[str, Path]):
        """Load tokenizer from directory."""
        path = Path(path)
        tokenizer_file = path / "tokenizer.json"
        config_file = path / "tokenizer_config.json"
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))

        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            if "special_tokens" in config:
                self._special_token_map.update(config["special_tokens"])

            # Load chat template from config
            if "chat_template" in config:
                self.set_chat_template(config["chat_template"])

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "AutoTokenizer":
        """Load tokenizer from pretrained directory.

        Raises:
            FileNotFoundError: If tokenizer.json is missing.
            RuntimeError: If tokenizer failed to initialize.
        """
        path = Path(path)
        tokenizer_file = path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found: {tokenizer_file}. "
                "A valid tokenizer.json is required."
            )
        instance = cls(path)
        if instance._tokenizer is None:
            raise RuntimeError(
                f"Failed to load tokenizer from {path}. "
                "The tokenizer.json may be corrupted or incompatible."
            )
        return instance

    def save_pretrained(self, save_path: str):
        """
        Save tokenizer to pretrained directory.

        Args:
            save_path: Path to save the tokenizer
        """

        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer not initialized. Load or create a tokenizer first."
            )

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save tokenizer
        self._tokenizer.save(str(save_path / "tokenizer.json"))

        # Save tokenizer config
        config = {}
        if self._special_token_map is not None:
            config["special_tokens"] = self._special_token_map
        if self._chat_template is not None:
            config["chat_template"] = self._chat_template.template_str

        with open(save_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def register_tokenizer(cls, name: str, tokenizer_class: type):
        """
        Register a new tokenizer class.

        Args:
            name: Name to register the tokenizer class under
            tokenizer_class: The tokenizer class to register
        """
        cls.TOKENIZER_CLASSES[name] = tokenizer_class

    def encode(
        self,
        tokens: Union[str, List[str]],
        out_ids: bool = True,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> List:
        """Encode text to tokens or token IDs."""
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
        else:
            encoded_list = self._tokenizer.encode_batch(
                tokens,
                is_pretokenized=is_pretokenized,
                add_special_tokens=add_special_tokens,
            )
            return [
                encoded.ids if out_ids else encoded.tokens for encoded in encoded_list
            ]

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
        """
        Dynamically intercept special token attribute access.
        Supports three forms:
          - tokenizer.bos_token   → returns string
          - tokenizer.bos_token_id → returns corresponding integer ID
          - tokenizer.stop_ids → returns list of corresponding integer IDs for all special tokens

        Internal/private attrs are not intercepted: during unpickle
        ``__dict__`` is empty, so probing ``self._special_token_map``
        would recurse infinitely.
        """
        if key.startswith("_"):
            raise AttributeError(key)

        # Handle stop_ids - return IDs for all special tokens
        if key == "stop_ids":
            stop_ids = []

            if self._tokenizer is None:
                return stop_ids

            for val in self._special_token_map.values():
                token_id = self._tokenizer.token_to_id(val)
                if token_id is not None:
                    stop_ids.append(token_id)

            return stop_ids

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
    def vocab_size(self) -> int:
        return len(self)

    def set_chat_template(self, template: Union[str, ChatTemplate]):
        """
        Set the chat template for the tokenizer.

        Args:
            template: Either a template name (str) registered in the global registry,
                      or a ChatTemplate instance, or a Jinja2 template string.

        Raises:
            KeyError: If template name is not registered.
        """
        if isinstance(template, str):
            self._chat_template = ChatTemplate.from_string(template)
        elif isinstance(template, ChatTemplate):
            self._chat_template = template
        else:
            raise ValueError("Invalid template type, must be str or ChatTemplate.")

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> Union[str, List[int]]:
        """
        Apply the chat template to messages and optionally tokenize the result.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            system_prompt: Optional system prompt string (auto-converted to first message).
            tokenize: Whether to return token IDs (True) or raw string (False).
            add_generation_prompt: Whether to add the generation prompt (default: True).
            **kwargs: Additional variables to pass to the template.

        Returns:
            Either the rendered string or list of token IDs.

        Raises:
            RuntimeError: If chat template is not set.
        """
        if self._chat_template is None:
            raise RuntimeError(
                "Chat template not set. Use set_chat_template() to set a template first."
            )

        # Auto-convert system_prompt to first message if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + list(messages)

        # Render the template
        rendered = self._chat_template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

        if tokenize:
            return self.encode(rendered)

        return rendered
