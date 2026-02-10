# Build stage for Linux (no Metal support)
FROM golang:1.23-alpine AS linux-builder

# Install build dependencies
RUN apk add --no-cache build-base

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application for Linux (CPU/SIMD only)
RUN CGO_ENABLED=0 GOOS=linux go build -o quarrel-linux ./cmd/simple

# Build stage for Linux with CUDA support (requires NVIDIA Container Toolkit)
FROM nvidia/cuda:12.4-devel-ubuntu22.04 AS cuda-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    golang-1.25 git build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    GO111MODULE=on

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

ENV CGO_LDFLAGS="-L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu -lcudart -lcuda -lcublas"
ENV CGO_CFLAGS="-I/usr/local/cuda/include"

RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -tags cuda \
    -ldflags "-linkmode=external -extldflags=-static" \
    -o quarrel-cuda ./cmd/cuda

# Build stage for macOS (with Metal/CGO support)
FROM --platform=linux/amd64 golang:1.23-alpine AS darwin-builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application for macOS (with Metal/CGO support)
RUN CGO_ENABLED=1 GOOS=darwin CC=o64-clang go build -o quarrel-darwin ./cmd/simple

# Final stage
FROM alpine:latest

WORKDIR /app

# Copy Linux binary (default)
COPY --from=linux-builder /app/quarrel-linux ./quarrel

# Set entrypoint
ENTRYPOINT ["./quarrel"]

# =============================================================================
# BUILD INSTRUCTIONS
# =============================================================================
#
# Linux (CPU/SIMD):
#   docker build -t longbow-quarrel:linux .
#   docker run -v $(pwd)/models:/data longbow-quarrel:linux --model /data/model.gguf
#
# Linux (CUDA):
#   docker build -f Dockerfile.cuda -t longbow-quarrel:cuda .
#   docker run --gpus all -v $(pwd)/models:/data longbow-quarrel:cuda --model /data/model.gguf
#
# macOS (Metal):
#   docker build --platform=linux/amd64 -t longbow-quarrel:daily .
#   # Transfer quarrel-darwin binary to macOS and run directly
#
# =============================================================================
