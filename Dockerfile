# AstrAI Dockerfile - Multi-stage Build (Optimized)
#
# CUDA version selection:
#   docker build -t astrai .
#   docker build -t astrai --build-arg CUDA_TAG=cu128 .
#   docker build -t astrai --build-arg CUDA_TAG=cu130 .
# Default: cu128

# Build stage - use base image with minimal build tools
FROM ubuntu:24.04 AS builder

ARG CUDA_TAG=cu128

WORKDIR /app

# Install Python 3.12 and minimal build dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python3.12 -m venv --copies /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code and install (deps read from pyproject.toml)
COPY astrai/ ./astrai/
COPY csrc/ ./csrc/
COPY setup.py .
COPY pyproject.toml .
COPY README.md .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# Production stage
FROM ubuntu:24.04 AS production

WORKDIR /app

# Install Python 3.12 runtime and healthcheck dependency
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.12 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY astrai/ ./astrai/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY pyproject.toml .
COPY README.md .

# Create non-root user matching the host uid/gid (passed via build args).
# ubuntu:24.04 ships a default 'ubuntu' user/group at uid/gid 1000, so remove
# it first to free those ids before creating astrai.
ARG USER_UID=1000
ARG USER_GID=1000
RUN userdel -r ubuntu 2>/dev/null || true \
    && groupdel ubuntu 2>/dev/null || true \
    && groupadd -g "${USER_GID}" astrai \
    && useradd -m -u "${USER_UID}" -g astrai astrai \
    && chown -R astrai:astrai /app
ENV HOME=/home/astrai
USER astrai

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1