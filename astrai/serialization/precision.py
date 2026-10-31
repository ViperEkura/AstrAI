"""Export weight formats without changing resumable training state."""

import torch


def _tensor_identity(tensor):
    return tensor.device, tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride())


def export_model_state(state_dict, dtype, master_parameters=()):
    if dtype is None:
        return state_dict
    target = getattr(torch, dtype)
    sources = {
        _tensor_identity(model): master.detach() for model, master in master_parameters
    }
    result = {}
    for name, value in state_dict.items():
        source = sources.get(_tensor_identity(value), value)
        result[name] = source.to(
            device="cpu",
            dtype=target if source.is_floating_point() else source.dtype,
            copy=True,
        )
    return result
