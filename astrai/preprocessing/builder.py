"""Mask building for preprocessing pipeline.

:class:`SectionRenderer` converts section specs into token ids and loss
masks (template / text / value extraction).  :class:`SingleOutputMaskBuilder`
handles single-output (SFT / pretrain), :class:`MultiOutputMaskBuilder`
handles multi-output (DPO / GRPO), and :class:`SectionedMaskBuilder`
orchestrates both modes as a façade.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


def _extract_domain(item: Dict, domain_key: Optional[str]) -> str:
    if not domain_key:
        return "__default__"
    val = item.get(domain_key, "__default__")
    return val if isinstance(val, str) else "__default__"


def _resolve_action(action: str, role: str, config) -> str:
    if action == "$role":
        return config.mask.get(role, config.mask_default)
    return action


class SectionRenderer:
    """Render section specs into ``(ids, loss_mask)`` tuples."""

    def _plan_sections(
        self, item, sections, config, tokenizer, *, filter_text: bool
    ) -> List[Tuple[str, str, bool]]:
        """Resolve section semantics before choosing scalar or batch encoding."""
        plan: List[Tuple[str, str, bool]] = []
        first_section = True
        for sec in sections:
            field = sec["field"]
            action = sec["action"]
            use_template = sec.get("template", False)
            if use_template:
                messages = item.get(field)
                if not isinstance(messages, list) or not messages:
                    continue
                for msg in messages:
                    rendered = tokenizer.apply_chat_template(
                        [msg], tokenize=False, add_generation_prompt=False
                    )
                    plan.append(
                        (
                            rendered,
                            _resolve_action(action, msg.get("role", ""), config),
                            False,
                        )
                    )
            else:
                text = str(item.get(field, ""))
                if not text.strip():
                    continue
                if filter_text:
                    pp = config.preprocessing
                    if len(text) < pp.min_chars or len(text) > pp.max_chars:
                        continue
                plan.append(
                    (text, action, sec.get("add_special_tokens", first_section))
                )
            first_section = False
        return plan

    @staticmethod
    def _check_ids(ids: List[int]) -> List[int]:
        if not isinstance(ids, list) or any(
            not isinstance(token, int) for token in ids
        ):
            raise ValueError("tokenizer.encode must return a list of integer token IDs")
        return ids

    def _encode_plans(
        self, plans: List[List[Tuple[str, str, bool]]], tokenizer, *, batched: bool
    ) -> List[List[List[int]]]:
        if not batched:
            return [
                [
                    self._check_ids(tokenizer.encode(text, add_special_tokens=add))
                    for text, _, add in plan
                ]
                for plan in plans
            ]

        encoded: List[List[List[int]]] = [[[] for _ in plan] for plan in plans]
        for add_special in (False, True):
            refs = [
                (item_idx, unit_idx, text)
                for item_idx, plan in enumerate(plans)
                for unit_idx, (text, _, add) in enumerate(plan)
                if add == add_special
            ]
            if not refs:
                continue
            ids_batch = tokenizer.encode(
                [text for _, _, text in refs], add_special_tokens=add_special
            )
            if not isinstance(ids_batch, list) or len(ids_batch) != len(refs):
                raise ValueError(
                    "tokenizer.encode batch must return one ID list per text"
                )
            for (item_idx, unit_idx, _), ids in zip(refs, ids_batch, strict=True):
                encoded[item_idx][unit_idx] = self._check_ids(ids)
        return encoded

    @staticmethod
    def _assemble_sections(plan, encoded, config, tokenizer, *, prepend_bos: bool):
        all_ids: List[int] = []
        loss_mask: List[int] = []
        if prepend_bos and tokenizer.bos_token_id is not None:
            all_ids.append(tokenizer.bos_token_id)
            loss_mask.append(0)
        for (_, action, _), ids in zip(plan, encoded, strict=True):
            all_ids.extend(ids)
            loss_mask.extend([1 if action == "train" else 0] * len(ids))
        max_len = config.preprocessing.max_seq_len
        all_ids = all_ids[:max_len]
        loss_mask = loss_mask[: len(all_ids)]
        if not all_ids or (prepend_bos and len(all_ids) <= 1):
            return None, None
        return all_ids, loss_mask

    def _render_sections(
        self,
        items,
        sections,
        config,
        tokenizer,
        *,
        batched: bool,
        is_top_level=False,
        filter_text=True,
    ):
        has_template = any(s.get("template") for s in sections)
        filter_text = (
            filter_text
            and not has_template
            and all(s["action"] == "train" for s in sections)
        )
        plans = [
            self._plan_sections(
                item, sections, config, tokenizer, filter_text=filter_text
            )
            for item in items
        ]
        encoded = self._encode_plans(plans, tokenizer, batched=batched)
        return [
            self._assemble_sections(
                plan, ids, config, tokenizer, prepend_bos=is_top_level and has_template
            )
            for plan, ids in zip(plans, encoded, strict=True)
        ]

    def process_sections(
        self,
        item: Dict,
        sections: List,
        config,
        tokenizer,
        *,
        is_top_level: bool = False,
    ):
        return self._render_sections(
            [item],
            sections,
            config,
            tokenizer,
            batched=False,
            is_top_level=is_top_level,
        )[0]

    def process_sections_batch(
        self,
        items: List[Dict],
        sections: List,
        config,
        tokenizer,
        *,
        is_top_level=False,
        filter_text=True,
    ):
        """Render a group while batching tokenization by special-token policy."""
        return self._render_sections(
            items,
            sections,
            config,
            tokenizer,
            batched=True,
            is_top_level=is_top_level,
            filter_text=filter_text,
        )

    def process_list_field(self, item: Dict, sections: List, config, tokenizer):
        """Tokenize a list-valued field, preserving per-element boundaries.

        Returns ``(list_of_id_lists, list_of_mask_lists)`` where each
        inner list corresponds to one element of the source list.  This
        is critical for GRPO where each response must stay a separate
        sequence so the strategy can form a ``[G, R]`` tensor.
        """
        return self._render_list_field(
            [item], sections, config, tokenizer, batched=False
        )[0]

    def process_list_field_batch(self, items, sections, config, tokenizer):
        return self._render_list_field(items, sections, config, tokenizer, batched=True)

    def _render_list_field(self, items, sections, config, tokenizer, *, batched: bool):
        per_item_ids = [[] for _ in items]
        per_item_masks = [[] for _ in items]

        for sec in sections:
            wrappers = []
            owners = []
            field = sec["field"]
            for item_idx, item in enumerate(items):
                values = item.get(field)
                if not isinstance(values, list):
                    continue
                for val in values:
                    if sec.get("template", False) and not isinstance(val, list):
                        continue
                    wrappers.append({field: val if isinstance(val, list) else str(val)})
                    owners.append(item_idx)

            # Each list element is an independent sequence. Unlike ordinary
            # first sections, it adds no special tokens unless requested.
            section = {"add_special_tokens": False, **sec}
            if batched:
                rendered = self.process_sections_batch(
                    wrappers,
                    [section],
                    config,
                    tokenizer,
                    is_top_level=False,
                    filter_text=False,
                )
            else:
                rendered = self._render_sections(
                    wrappers,
                    [section],
                    config,
                    tokenizer,
                    batched=False,
                    filter_text=False,
                )
            for owner, (ids, mask) in zip(owners, rendered, strict=True):
                if ids:
                    per_item_ids[owner].append(ids)
                    per_item_masks[owner].append(mask)

        return [
            (ids, masks) if ids else (None, None)
            for ids, masks in zip(per_item_ids, per_item_masks)
        ]

    @staticmethod
    def is_value_section(sections: List) -> bool:
        return len(sections) == 1 and sections[0].get("action") == "value"

    @staticmethod
    def extract_raw_value(item: Dict, sections: List):
        sec = sections[0]
        field = sec["field"]
        raw = item.get(field)
        if raw is None:
            return None
        if isinstance(raw, list):
            return [float(v) for v in raw]
        return [float(raw)]


class BaseMaskBuilder(ABC):
    """Convert a JSONL item into token ids and optional loss_mask."""

    @abstractmethod
    def build(self, item: Dict, config, tokenizer) -> Optional[Dict]: ...

    def build_batch(self, items: List[Dict], config, tokenizer) -> List[Optional[Dict]]:
        return [self.build(item, config, tokenizer) for item in items]


class SingleOutputMaskBuilder(BaseMaskBuilder):
    """Build a single output sequence with optional loss mask.

    Expects ``config.input.sections`` (list of section specs).
    """

    def __init__(self, renderer: Optional[SectionRenderer] = None):
        self.renderer = renderer or SectionRenderer()

    def build(self, item: Dict, config, tokenizer) -> Optional[Dict]:
        sections = config.input.sections
        if not sections:
            return None
        rendered = self.renderer.process_sections(
            item, sections, config, tokenizer, is_top_level=True
        )
        return self._assemble_outputs([item], [rendered], config)[0]

    def build_batch(self, items, config, tokenizer):
        sections = config.input.sections
        if not sections:
            return [None] * len(items)
        rendered = self.renderer.process_sections_batch(
            items, sections, config, tokenizer, is_top_level=True
        )
        return self._assemble_outputs(items, rendered, config)

    @staticmethod
    def _assemble_outputs(items, rendered, config):
        results = []
        for item, (ids, mask) in zip(items, rendered, strict=True):
            if ids is None:
                results.append(None)
                continue
            result = {
                "sequence": ids,
                "domain": _extract_domain(item, config.output.domain_key),
            }
            if not all(m == 1 for m in mask):
                result["loss_mask"] = mask
            results.append(result)
        return results


class MultiOutputMaskBuilder(BaseMaskBuilder):
    """Build multiple output sequences (DPO / GRPO).

    Expects ``config.input.sources`` (dict of output_key → spec).
    """

    def __init__(self, renderer: Optional[SectionRenderer] = None):
        self.renderer = renderer or SectionRenderer()

    def build(self, item: Dict, config, tokenizer) -> Optional[Dict]:
        return self._build_outputs([item], config, tokenizer, batched=False)[0]

    def build_batch(self, items, config, tokenizer):
        return self._build_outputs(items, config, tokenizer, batched=True)

    def _build_outputs(self, items, config, tokenizer, *, batched: bool):
        sources_spec = getattr(config.input, "sources", None) or {}
        results = [{} for _ in items]
        sources = {
            output_key: spec
            for output_key, spec in sources_spec.items()
            if spec.get("sections")
        }
        for output_key, spec in sources.items():
            sections = spec["sections"]
            if self.renderer.is_value_section(sections):
                rendered = [
                    (self.renderer.extract_raw_value(item, sections), None)
                    for item in items
                ]
            elif spec.get("list_field", False) and batched:
                rendered = self.renderer.process_list_field_batch(
                    items, sections, config, tokenizer
                )
            elif batched:
                rendered = self.renderer.process_sections_batch(
                    items, sections, config, tokenizer, is_top_level=True
                )
            elif spec.get("list_field", False):
                rendered = [
                    self.renderer.process_list_field(item, sections, config, tokenizer)
                    for item in items
                ]
            else:
                rendered = [
                    self.renderer.process_sections(
                        item, sections, config, tokenizer, is_top_level=True
                    )
                    for item in items
                ]

            for result, (ids, mask) in zip(results, rendered, strict=True):
                if ids is None:
                    continue
                result[output_key] = ids
                if mask is not None and (
                    spec.get("list_field", False)
                    or "mask_key" in spec
                    or not all(m == 1 for m in mask)
                ):
                    mask_key = spec.get("mask_key", f"{output_key}_mask")
                    result[mask_key] = mask

        return [
            ({**result, "domain": _extract_domain(item, config.output.domain_key)})
            if sources and sources.keys() <= result.keys()
            else None
            for item, result in zip(items, results)
        ]


class SectionedMaskBuilder(BaseMaskBuilder):
    """Façade that dispatches to SingleOutputMaskBuilder or MultiOutputMaskBuilder.

    Preserves backward compatibility for existing configs and code that rely
    on the ``"sectioned"`` factory name.
    """

    def __init__(self):
        self._single = SingleOutputMaskBuilder()
        self._multi = MultiOutputMaskBuilder()

    def build(self, item: Dict, config, tokenizer) -> Optional[Dict]:
        sources_spec = getattr(config.input, "sources", None)
        if sources_spec:
            return self._multi.build(item, config, tokenizer)
        return self._single.build(item, config, tokenizer)

    def build_batch(self, items, config, tokenizer):
        sources_spec = getattr(config.input, "sources", None)
        if sources_spec:
            return self._multi.build_batch(items, config, tokenizer)
        return self._single.build_batch(items, config, tokenizer)
