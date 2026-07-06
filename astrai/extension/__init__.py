import importlib
import logging

logger = logging.getLogger(__name__)

available: dict[str, bool] = {}

for _name in ["gqa_decode_attn", "gqa_prefill_attn"]:
    try:
        importlib.import_module(f".{_name}", package=__package__)
        available[_name] = True
    except ImportError:
        available[_name] = False
