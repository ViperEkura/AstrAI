"""Control-plane messages for the local online rollout worker processes.

Each worker has one ordered, duplex pipe and at most one outstanding command.
Large policy tensors live in shared memory; only metadata and CPU rollout
results are serialized through these messages.
"""

from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.connection import Connection
from typing import Any

from astrai.trainer.rollout.types import RawRollout

PROTOCOL_VERSION = 1


class MessageKind(StrEnum):
    READY = "ready"
    WEIGHT = "weight"
    WEIGHT_ACK = "weight_ack"
    GENERATE = "generate"
    RESULT = "result"
    ERROR = "error"
    STOP = "stop"


class RolloutProtocolError(RuntimeError):
    """A worker sent a malformed or out-of-order control message."""


@dataclass(frozen=True, slots=True)
class RolloutMessage:
    kind: MessageKind
    request_id: int
    round_id: int | None = None
    policy_version: int | None = None
    payload: Any = None
    protocol_version: int = PROTOCOL_VERSION

    def expect(
        self,
        kind: MessageKind,
        request_id: int,
        *,
        round_id: int | None = None,
        policy_version: int | None = None,
    ) -> None:
        """Validate a reply against its one outstanding worker command."""
        if self.kind != kind:
            raise RolloutProtocolError(f"unexpected rollout message {self.kind}")
        if self.request_id != request_id:
            raise RolloutProtocolError("wrong request ID")
        if round_id is not None and self.round_id != round_id:
            raise RolloutProtocolError("stale round ID")
        if policy_version is not None and self.policy_version != policy_version:
            raise RolloutProtocolError("wrong policy version")
        expected_payload = {
            MessageKind.READY: WorkerReady,
            MessageKind.WEIGHT_ACK: WeightAck,
            MessageKind.RESULT: GenerationResult,
        }[kind]
        if not isinstance(self.payload, expected_payload):
            raise RolloutProtocolError("invalid rollout payload")


@dataclass(frozen=True, slots=True)
class WorkerReady:
    cuda_graph_enabled: bool
    peak_gpu_memory: int
    shared_memory_pinned: bool


@dataclass(frozen=True, slots=True)
class WeightAck:
    peak_gpu_memory: int


@dataclass(frozen=True, slots=True)
class GenerationResult:
    rollout: RawRollout
    generation_seconds: float
    peak_gpu_memory: int


def send_message(conn: Connection, message: RolloutMessage) -> None:
    if not isinstance(message, RolloutMessage):
        raise RolloutProtocolError("rollout message has an invalid envelope")
    conn.send(message)


def recv_message(conn: Connection) -> RolloutMessage:
    message = conn.recv()
    if not isinstance(message, RolloutMessage):
        raise RolloutProtocolError("rollout message has an invalid envelope")
    if message.protocol_version != PROTOCOL_VERSION:
        raise RolloutProtocolError(
            f"rollout protocol version {message.protocol_version} is unsupported"
        )
    if not isinstance(message.kind, MessageKind):
        raise RolloutProtocolError("rollout message has an unknown kind")
    if type(message.request_id) is not int or message.request_id < 0:
        raise RolloutProtocolError("rollout message has an invalid request ID")
    return message
