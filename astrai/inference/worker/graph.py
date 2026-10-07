"""CUDA-graph capture for the decode model-forward step.

Mirrors SGLang's cuda-graph manager: one graph per batch size.  The graph
pair.  The graph captures ``model.forward()`` with workspace-backed inputs
(all at fixed addresses).  Before each replay the caller updates the input
buffer content in-place so the graph sees fresh data at the same tensor
addresses.

Only the model forward is captured — sampling runs outside the graph
(via ``torch.multinomial`` which consumes a mutable RNG state).
"""

import torch
from torch import Tensor


class CUDAGraphRunner:
    """CUDA-graph capture/replay for decode steps.

    Parameters:
        enabled: When ``False``, ``forward()`` always runs the live model
            forward without capture/replay (graphs are cleared).  Toggle at
            runtime via the ``set_enabled()`` method.

    Usage::

        gctx = CUDAGraphRunner()
        with torch.inference_mode():
            outputs = gctx.forward(
                model,
                key=(batch_size,),
                input_ids=workspace.input_ids[:b].unsqueeze(1),
                input_mask=input_mask,
                kv_cache=kv_cache,
                position_ids=workspace.position_ids[:b].unsqueeze(1),
            )

    The first call at a given key runs *without* capture (warmup).  The
    second call captures the graph.  Subsequent calls replay the captured
    graph.  Capture uses an explicit stream on the input tensor's device:
    PyTorch's implicit graph stream can remain bound to the first GPU used
    by a process, leaving later GPUs with an empty graph.
    """

    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self._graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self._outputs: dict[tuple, dict[str, Tensor]] = {}
        self._warmed: set[tuple] = set()
        self._capture_stream: torch.cuda.Stream | None = None
        self._captures_frozen = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, flag: bool):
        """Enable or disable CUDA-graph capture at runtime.

        Disabling clears all captured graphs (frees GPU memory) and warmup
        state.  Re-enabling after disable starts fresh — graphs are
        re-captured on the next warmup cycle.
        """
        if flag == self._enabled:
            return
        self._enabled = flag
        if not flag:
            self._graphs.clear()
            self._outputs.clear()
            self._warmed.clear()
            self._capture_stream = None
            self._captures_frozen = False

    def freeze_captures(self):
        """Use eager forward for uncaptured shapes after startup warmup.

        A process can only capture one CUDA graph at a time.  Multi-GPU
        rollout workers may replay concurrently, but must never initiate a
        new capture while another worker is generating.
        """
        self._captures_frozen = True

    def forward(self, model, *, key, **kwargs) -> dict[str, Tensor]:
        """Run ``model(**kwargs)`` via graph replay or live forward.

        Args:
            model: callable, e.g. ``self.model.forward``.
            key: ``(batch_size,)`` — the dispatch key (one graph per batch size).
            **kwargs: arguments forwarded to ``model``.  All tensor arguments
                must reside at stable addresses (workspace buffers).

        Returns:
            The dict produced by ``model(**kwargs)``, e.g.
            ``{"logits": ..., "h0": ...}``.
        """
        if not self._enabled:
            self._outputs[key] = model(**kwargs)
            return self._outputs[key]

        if key in self._graphs:
            self._graphs[key].replay()
        elif self._captures_frozen:
            return model(**kwargs)
        elif key in self._warmed:
            cap_output = model(**kwargs)
            device = next(
                value.device for value in kwargs.values() if isinstance(value, Tensor)
            )
            torch.cuda.synchronize(device)
            if self._capture_stream is None:
                self._capture_stream = torch.cuda.Stream(device=device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self._capture_stream):
                self._outputs[key] = model(**kwargs)
            self._graphs[key] = graph
            self._warmed.discard(key)
            return cap_output
        else:
            self._warmed.add(key)
            self._outputs[key] = model(**kwargs)
        return self._outputs[key]

    def has_graph(self, key: tuple) -> bool:
        return key in self._graphs
