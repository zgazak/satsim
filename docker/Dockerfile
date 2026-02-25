# Multi-stage Dockerfile for SatSim
# CPU by default, GPU via: docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE} AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY README.md .

# Install package
RUN pip install --no-cache-dir -e ".[dev]"

# Default command
CMD ["satsim", "--help"]
