# syntax=docker/dockerfile:1.4

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM --platform=$BUILDPLATFORM golang:1.21-alpine AS builder

ARG TARGETOS
ARG TARGETARCH
ARG TARGETVARIANT

WORKDIR /src

RUN apk add --no-cache ca-certificates git build-base

# Copy go.mod and go.sum first for better caching
COPY go.mod go.sum* ./

RUN go mod download

# Copy the rest of the source code
COPY . .

# Build the webui binary with CUDA support
RUN CGO_ENABLED=1 GOOS=$TARGETOS GOARCH=$TARGETARCH \
    go build -tags "webui,cuda" \
    -ldflags="-s -w -linkmode=external -extldflags=-static" \
    -o /out/webui ./cmd/webui

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM --platform=$TARGETPLATFORM nvidia/cuda:12.2-runtime-ubuntu22.04 AS runtime

ARG WEBUI_VERSION=dev

LABEL maintainer="Longbow-Quarrel <dev@longbow.example.com>"
LABEL description="Longbow-Quarrel WebUI - LLM Inference Engine Web Interface"
LABEL version="${WEBUI_VERSION}"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libcudnn8 \
    libcudnn8-dev \
    libnccl2 \
    libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r quarrel && useradd -r -g quarrel quarrel && \
    mkdir /home/quarrel && chown quarrel:quarrel /home/quarrel

# Set working directory
WORKDIR /home/quarrel

# Copy binary from builder
COPY --from=builder /out/webui /usr/local/bin/webui

# Copy static assets
COPY --from=builder --chown=quarrel:quarrel /src/cmd/webui/static ./static
COPY --from=builder --chown=quarrel:quarrel /src/cmd/webui/templates ./templates

# Create directories for models and data
RUN mkdir -p /data/models /data/cache && \
    chown -R quarrel:quarrel /data

# Expose ports
EXPOSE 8080 9090

# Set environment variables
ENV WEBUI_PORT=8080
ENV WEBUI_METRICS_PORT=9090
ENV WEBUI_HOST=0.0.0.0
ENV WEBUI_DATA_DIR=/data
ENV WEBUI_CACHE_DIR=/data/cache

# Switch to non-root user
USER quarrel

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${WEBUI_PORT}/healthz || exit 1

# Entrypoint
ENTRYPOINT ["webui"]

# Default command
CMD ["--port", "${WEBUI_PORT}", "--metrics-port", "${WEBUI_METRICS_PORT}", "--host", "${WEBUI_HOST}"]

# =============================================================================
# Stage 3: CPU-only Runtime (alternative)
# =============================================================================
FROM --platform=$TARGETPLATFORM alpine:3.19 AS runtime-cpu

ARG WEBUI_VERSION=dev

LABEL maintainer="Longbow-Quarrel <dev@longbow.example.com>"
LABEL description="Longbow-Quarrel WebUI - LLM Inference Engine Web Interface (CPU-only)"
LABEL version="${WEBUI_VERSION}"

# Install runtime dependencies
RUN apk add --no-cache ca-certificates curl

# Create non-root user
RUN addgroup -S quarrel && adduser -S quarrel -G quarrel && \
    mkdir /home/quarrel && chown quarrel:quarrel /home/quarrel

WORKDIR /home/quarrel

# Copy binary from builder
COPY --from=builder /out/webui /usr/local/bin/webui

# Copy static assets
COPY --from=builder --chown=quarrel:quarrel /src/cmd/webui/static ./static
COPY --from=builder --chown=quarrel:quarrel /src/cmd/webui/templates ./templates

# Create directories for models and data
RUN mkdir -p /data/models /data/cache && \
    chown -R quarrel:quarrel /data

# Expose ports
EXPOSE 8080 9090

# Set environment variables
ENV WEBUI_PORT=8080
ENV WEBUI_METRICS_PORT=9090
ENV WEBUI_HOST=0.0.0.0
ENV WEBUI_DATA_DIR=/data
ENV WEBUI_CACHE_DIR=/data/cache

# Switch to non-root user
USER quarrel

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${WEBUI_PORT}/healthz || exit 1

# Entrypoint
ENTRYPOINT ["webui"]

# Default command
CMD ["--port", "${WEBUI_PORT}", "--metrics-port", "${WEBUI_METRICS_PORT}", "--host", "${WEBUI_HOST}"]
