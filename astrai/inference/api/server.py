"""
OpenAI / Anthropic-compatible chat completion server backed by continuous-batching inference.

Protocol-specific formatting is delegated to ``astrai.inference.protocol``.
This module owns the FastAPI app, request/response schemas, and dependency wiring.

``app`` is lazily constructed — importing this module does NOT create a FastAPI instance.
Use :func:`get_app` to access the singleton.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from astrai.inference.api.anthropic import AnthropicResponseBuilder
from astrai.inference.api.openai import OpenAIResponseBuilder
from astrai.inference.api.protocol import ProtocolHandler
from astrai.inference.engine import InferenceEngine
from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer

logger = logging.getLogger(__name__)

_app_instance: Optional[FastAPI] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completion API request body."""

    model: str = "astrai"
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=2048, ge=1)
    n: Optional[int] = Field(default=1, ge=1)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[int, float]] = None
    user: Optional[str] = None


class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class MessagesRequest(BaseModel):
    """Anthropic Messages API request body."""

    model: str = "astrai"
    max_tokens: int = Field(default=1024, ge=1)
    messages: List[AnthropicMessage]
    system: Optional[str] = None
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=1)
    stream: Optional[bool] = False
    stop_sequences: Optional[List[str]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = app.state.server_config
    if not config.get("_test", False):
        try:
            app.state.engine = _create_engine(**config)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    yield
    if app.state.engine:
        app.state.engine.shutdown()
        logger.info("Inference engine shutdown complete")


router = APIRouter()


def _create_engine(
    param_path: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_batch_size: int = 16,
) -> InferenceEngine:
    if not param_path.exists():
        raise FileNotFoundError(f"Parameter directory not found: {param_path}")

    tokenizer = AutoTokenizer.from_pretrained(param_path)
    model = AutoModel.from_pretrained(param_path)
    model.to(device=device, dtype=dtype)
    logger.info(f"Model loaded on {device} with dtype {dtype}")

    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
    )
    logger.info(f"Inference engine initialized with max_batch_size={max_batch_size}")
    return engine


def get_app() -> FastAPI:
    """Return the singleton FastAPI instance (lazily created on first call)."""
    global _app_instance
    if _app_instance is None:
        _app_instance = FastAPI(
            title="AstrAI Inference Server",
            version="0.2.0",
            lifespan=lifespan,
        )
        _app_instance.include_router(router)
        _app_instance.state.server_config = {}
        _app_instance.state.engine = None
    return _app_instance


def _get_engine() -> InferenceEngine:
    engine = get_app().state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine


@router.get("/health")
async def health():
    app = get_app()
    return {
        "status": "ok",
        "model_loaded": app.state.engine is not None,
    }


@router.get("/stats")
async def get_stats():
    return _get_engine().get_stats()


@router.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    engine = _get_engine()
    handler = ProtocolHandler(request, engine, OpenAIResponseBuilder())
    return await handler.handle()


@router.post("/v1/messages")
async def create_message(request: MessagesRequest):
    engine = _get_engine()
    handler = ProtocolHandler(request, engine, AnthropicResponseBuilder())
    return await handler.handle()


def run_server(
    param_path: Path,
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_batch_size: int = 16,
):
    app = get_app()
    app.state.server_config = {
        "device": device,
        "dtype": dtype,
        "param_path": param_path,
        "max_batch_size": max_batch_size,
    }
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
    )
