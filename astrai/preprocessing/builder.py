"""Mask building for preprocessing pipeline.

:class:`SectionRenderer` converts section specs into token ids and loss
masks (template / text / value extraction).  :class:`SingleOutputMaskBuilder`
handles single-output (SFT / pretrain), :class:`MultiOutputMaskBuilder`
handles multi-output (DPO / GRPO), and :class:`SectionedMaskBuilder`
orchestrates both modes as a façade.
"""

from abc import ABC, abstractmethod
from typing import Optional

from astrai.factory import BaseFactory


def _extract_domain(item: dict, domain_key: Optional[str]) -> str:
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

    def process_sections(
        self,
        item: dict,
        sections: list,
        config,
        tokenizer,
        *,
        is_top_level: bool = False,
    ):
        all_ids: list[int] = []
        loss_mask: list[int] = []

        has_template = any(s.get("template") for s in sections)
        is_text_config = not has_template and all(
            s["action"] == "train" for s in sections
        )

        if is_top_level and has_template and tokenizer.bos_token_id is not None:
            all_ids.append(tokenizer.bos_token_id)
            loss_mask.append(0)

        first_section = True
        for sec in sections:
            field = sec["field"]
            action = sec["action"]
            use_template = sec.get("template", False)
            add_special = sec.get(
                "add_special_tokens", not use_template and first_section
            )

            if use_template:
                success = self._append_template(
                    item, field, action, tokenizer, config, all_ids, loss_mask
                )
                if not success:
                    continue
            else:
                success = self._append_text(
                    item,
                    field,
                    action,
                    tokenizer,
                    add_special,
                    is_text_config,
                    config,
                    all_ids,
                    loss_mask,
                )
                if not success:
                    continue

            first_section = False

        max_len = config.preprocessing.max_seq_len
        all_ids = all_ids[:max_len]
        loss_mask = loss_mask[: len(all_ids)]

        if not all_ids:
            return None, None

        if is_top_level and has_template and len(all_ids) <= 1:
            return None, None

        return all_ids, loss_mask

    def process_sections_batch(
        self,
        items: list[dict],
        sections: list,
        config,
        tokenizer,
        *,
        is_top_level=False,
        filter_text=True,
    ):
        """Render and tokenize a group of records with batched Rust tokenization."""
        has_template = any(s.get("template") for s in sections)
        is_text_config = not has_template and all(
            s["action"] == "train" for s in sections
        )
        plans: list[list[tuple[str, str, bool]]] = []

        for item in items:
            plan: list[tuple[str, str, bool]] = []
            first_section = True
            for sec in sections:
                field = sec["field"]
                action = sec["action"]
                use_template = sec.get("template", False)
                add_special = sec.get(
                    "add_special_tokens", not use_template and first_section
                )

                if use_template:
                    messages = item.get(field)
                    if not isinstance(messages, list) or not messages:
                        continue
                    for msg in messages:
                        role = msg.get("role", "")
                        rendered = tokenizer.apply_chat_template(
                            [msg], tokenize=False, add_generation_prompt=False
                        )
                        plan.append(
                            (rendered, _resolve_action(action, role, config), False)
                        )
                else:
                    text = str(item.get(field, ""))
                    if not text.strip():
                        continue
                    if is_text_config and filter_text:
                        pp = config.preprocessing
                        if pp.min_chars > 0 and len(text) < pp.min_chars:
                            continue
                        if len(text) > pp.max_chars:
                            continue
                    plan.append((text, action, add_special))

                first_section = False
            plans.append(plan)

        encoded: dict[tuple[int, int], list[int]] = {}
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
            for (item_idx, unit_idx, _), ids in zip(refs, ids_batch):
                encoded[(item_idx, unit_idx)] = ids

        outputs = []
        max_len = config.preprocessing.max_seq_len
        for item_idx, plan in enumerate(plans):
            all_ids = []
            loss_mask = []
            if is_top_level and has_template and tokenizer.bos_token_id is not None:
                all_ids.append(tokenizer.bos_token_id)
                loss_mask.append(0)
            for unit_idx, (_, action, _) in enumerate(plan):
                ids = encoded[(item_idx, unit_idx)]
                all_ids.extend(ids)
                loss_mask.extend([1 if action == "train" else 0] * len(ids))
            all_ids = all_ids[:max_len]
            loss_mask = loss_mask[: len(all_ids)]
            if not all_ids or (is_top_level and has_template and len(all_ids) <= 1):
                outputs.append((None, None))
            else:
                outputs.append((all_ids, loss_mask))
        return outputs

    def process_list_field(self, item: dict, sections: list, config, tokenizer):
        """Tokenize a list-valued field, preserving per-element boundaries.

        Returns ``(list_of_id_lists, list_of_mask_lists)`` where each
        inner list corresponds to one element of the source list.  This
        is critical for GRPO where each response must stay a separate
        sequence so the strategy can form a ``[G, R]`` tensor.
        """
        per_item_ids: list[list[int]] = []
        per_item_masks: list[list[int]] = []

        for sec in sections:
            field = sec["field"]
            action = sec["action"]
            use_template = sec.get("template", False)

            values = item.get(field)
            if not isinstance(values, list):
                continue

            for val in values:
                ids: list[int] = []
                mask: list[int] = []
                if use_template:
                    if isinstance(val, list):
                        wrapper = {field: val}
                        self._append_template(
                            wrapper, field, action, tokenizer, config, ids, mask
                        )
                else:
                    wrapper = {field: str(val)}
                    self._append_text(
                        wrapper,
                        field,
                        action,
                        tokenizer,
                        False,
                        False,
                        config,
                        ids,
                        mask,
                    )
                if ids:
                    max_len = config.preprocessing.max_seq_len
                    ids = ids[:max_len]
                    mask = mask[: len(ids)]
                    per_item_ids.append(ids)
                    per_item_masks.append(mask)

        if not per_item_ids:
            return None, None
        return per_item_ids, per_item_masks

    def process_list_field_batch(self, items, sections, config, tokenizer):
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

            rendered = self.process_sections_batch(
                wrappers,
                [sec],
                config,
                tokenizer,
                is_top_level=False,
                filter_text=False,
            )
            for owner, (ids, mask) in zip(owners, rendered):
                if ids:
                    per_item_ids[owner].append(ids)
                    per_item_masks[owner].append(mask)

        return [
            (ids, masks) if ids else (None, None)
            for ids, masks in zip(per_item_ids, per_item_masks)
        ]

    @staticmethod
    def is_value_section(sections: list) -> bool:
        return len(sections) == 1 and sections[0].get("action") == "value"

    @staticmethod
    def extract_raw_value(item: dict, sections: list):
        sec = sections[0]
        field = sec["field"]
        raw = item.get(field)
        if raw is None:
            return None
        if isinstance(raw, list):
            return [float(v) for v in raw]
        return [float(raw)]

    def _append_template(
        self, item, field, action, tokenizer, config, all_ids, loss_mask
    ):
        messages = item.get(field)
        if not isinstance(messages, list) or not messages:
            return False
        for msg in messages:
            role = msg.get("role", "")
            act = _resolve_action(action, role, config)
            rendered = tokenizer.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=False
            )
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            all_ids.extend(ids)
            val = 1 if act == "train" else 0
            loss_mask.extend([val] * len(ids))
        return True

    def _append_text(
        self,
        item,
        field,
        action,
        tokenizer,
        add_special,
        is_text_config,
        config,
        all_ids,
        loss_mask,
    ):
        text = str(item.get(field, ""))
        if not text.strip():
            return False
        if is_text_config:
            pp = config.preprocessing
            if pp.min_chars > 0 and len(text) < pp.min_chars:
                return False
            if len(text) > pp.max_chars:
                return False
        ids = tokenizer.encode(text, add_special_tokens=add_special)
        all_ids.extend(ids)
        val = 1 if action == "train" else 0
        loss_mask.extend([val] * len(ids))
        return True


class BaseMaskBuilder(ABC):
    """Convert a JSONL item into token ids and optional loss_mask."""

    @abstractmethod
    def build(self, item: dict, config, tokenizer) -> Optional[dict]: ...

    def build_batch(self, items: list[dict], config, tokenizer) -> list[Optional[dict]]:
        return [self.build(item, config, tokenizer) for item in items]


class MaskBuilderFactory(BaseFactory["BaseMaskBuilder"]):
    pass


@MaskBuilderFactory.register("single")
class SingleOutputMaskBuilder(BaseMaskBuilder):
    """Build a single output sequence with optional loss mask.

    Expects ``config.input.sections`` (list of section specs).
    """

    def __init__(self, renderer: Optional[SectionRenderer] = None):
        self.renderer = renderer or SectionRenderer()

    def build(self, item: dict, config, tokenizer) -> Optional[dict]:
        sections = config.input.sections
        if not sections:
            return None

        ids, mask = self.renderer.process_sections(
            item, sections, config, tokenizer, is_top_level=True
        )
        if ids is None:
            return None

        result: dict = {
            "sequence": ids,
            "domain": _extract_domain(item, config.output.domain_key),
        }
        if not all(m == 1 for m in mask):
            result["loss_mask"] = mask
        return result

    def build_batch(self, items, config, tokenizer):
        sections = config.input.sections
        if not sections:
            return [None] * len(items)
        rendered = self.renderer.process_sections_batch(
            items, sections, config, tokenizer, is_top_level=True
        )
        results = []
        for item, (ids, mask) in zip(items, rendered):
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


@MaskBuilderFactory.register("multi")
class MultiOutputMaskBuilder(BaseMaskBuilder):
    """Build multiple output sequences (DPO / GRPO).

    Expects ``config.input.sources`` (dict of output_key → spec).
    """

    def __init__(self, renderer: Optional[SectionRenderer] = None):
        self.renderer = renderer or SectionRenderer()

    def build(self, item: dict, config, tokenizer) -> Optional[dict]:
        sources_spec = getattr(config.input, "sources", None)
        if not sources_spec:
            return None

        result: dict = {}
        any_output = False

        for output_key, spec in sources_spec.items():
            sections = spec.get("sections", [])
            if not sections:
                continue

            if self.renderer.is_value_section(sections):
                ids = self.renderer.extract_raw_value(item, sections)
                if ids is None:
                    continue
                result[output_key] = ids
                any_output = True
                continue

            list_field = spec.get("list_field", False)
            mask_key = spec.get("mask_key", f"{output_key}_mask")

            if list_field:
                ids, mask = self.renderer.process_list_field(
                    item, sections, config, tokenizer
                )
                if ids is None:
                    continue
                # ids is List[List[int]] — preserve per-response structure
                result[output_key] = ids
                if mask is not None:
                    result[mask_key] = mask
                any_output = True
                continue

            ids, mask = self.renderer.process_sections(
                item, sections, config, tokenizer, is_top_level=True
            )

            if ids is None:
                continue

            result[output_key] = ids
            if not all(m == 1 for m in mask):
                result[mask_key] = mask
            elif "mask_key" in spec:
                result[mask_key] = mask

            any_output = True

        if not any_output:
            return None

        result["domain"] = _extract_domain(item, config.output.domain_key)
        return result

    def build_batch(self, items, config, tokenizer):
        sources_spec = getattr(config.input, "sources", None)
        if not sources_spec:
            return [None] * len(items)

        results = [{} for _ in items]
        for output_key, spec in sources_spec.items():
            sections = spec.get("sections", [])
            if not sections:
                continue
            if self.renderer.is_value_section(sections):
                for item, result in zip(items, results):
                    value = self.renderer.extract_raw_value(item, sections)
                    if value is not None:
                        result[output_key] = value
                continue

            mask_key = spec.get("mask_key", f"{output_key}_mask")
            if spec.get("list_field", False):
                rendered = self.renderer.process_list_field_batch(
                    items, sections, config, tokenizer
                )
            else:
                rendered = self.renderer.process_sections_batch(
                    items, sections, config, tokenizer, is_top_level=True
                )

            for result, (ids, mask) in zip(results, rendered):
                if ids is None:
                    continue
                result[output_key] = ids
                if spec.get("list_field", False) or not all(m == 1 for m in mask):
                    result[mask_key] = mask
                elif "mask_key" in spec:
                    result[mask_key] = mask

        return [
            ({**result, "domain": _extract_domain(item, config.output.domain_key)})
            if result
            else None
            for item, result in zip(items, results)
        ]


@MaskBuilderFactory.register("sectioned")
class SectionedMaskBuilder(BaseMaskBuilder):
    """Façade that dispatches to SingleOutputMaskBuilder or MultiOutputMaskBuilder.

    Preserves backward compatibility for existing configs and code that rely
    on the ``"sectioned"`` factory name.
    """

    def __init__(self):
        self._single = SingleOutputMaskBuilder()
        self._multi = MultiOutputMaskBuilder()

    def build(self, item: dict, config, tokenizer) -> Optional[dict]:
        sources_spec = getattr(config.input, "sources", None)
        if sources_spec:
            return self._multi.build(item, config, tokenizer)
        return self._single.build(item, config, tokenizer)

    def build_batch(self, items, config, tokenizer):
        sources_spec = getattr(config.input, "sources", None)
        if sources_spec:
            return self._multi.build_batch(items, config, tokenizer)
        return self._single.build_batch(items, config, tokenizer)
