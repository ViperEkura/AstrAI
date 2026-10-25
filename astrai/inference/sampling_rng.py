"""Request-local sampling draws, independent of scheduling and speculation."""

import hashlib
import json
import math
import struct

REQUEST_RNG_VERSION = "sha256-counter53-inverse-cdf-fp32-v1"
MAX_SEED = (1 << 63) - 1


def validate_seed(seed):
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError("sampling seed must be an integer in [0, 2**63-1]")


def request_seed(seed, policy_version, prompt_ids, response_index, occurrence=0):
    """Bind one stream to a prompt, response and immutable policy version."""
    validate_seed(seed)
    key = json.dumps(
        [seed, policy_version, list(prompt_ids), response_index, occurrence],
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & MAX_SEED


def sampling_uniform(seed, output_position):
    """Address a draw by output position; discarded work advances no state."""
    validate_seed(seed)
    if type(output_position) is not int or output_position < 0:
        raise ValueError("sampling output position must be nonnegative")
    key = b"astrai-request-sampling-v1\0" + struct.pack(">QQ", seed, output_position)
    bits = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 11
    return min((bits + 0.5) * 2**-53, math.nextafter(1.0, 0.0))
