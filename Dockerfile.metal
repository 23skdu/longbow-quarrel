# =============================================================================
# Longbow-Quarrel - Apple Metal Build
# =============================================================================
# This Dockerfile builds Quarrel with Metal acceleration for macOS arm64.
# Uses Apple Metal for GPU acceleration on Apple Silicon.
#
# Note: This build is designed to run on macOS (darwin/arm64).
# Build on Linux with cross-compiler or on macOS directly.
#
# Build:    docker build -f Dockerfile.metal -t longbow-quarrel:metal .
# Run:      docker run --device metal -v $(pwd)/models:/data longbow-quarrel:metal --model /data/model.gguf
# =============================================================================

# -----------------------------------------------------------------------------
# Build Stage: Metal Compilation (cross-compile from Linux amd64 to darwin arm64)
# -----------------------------------------------------------------------------
FROM --platform=linux/amd64 golang:1.27.0-alpine AS metal-builder

# Install build dependencies for cross-compilation
RUN apk add --no-cache build-base git clang llvm

# Cross-compile to Darwin arm64 with Metal support
# Note: Requires proper Darwin SDK and Metal headers for production builds
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build with CGO for Metal (darwin/arm64)
RUN CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 CC=o64-clang go build -o quarrel-metal ./cmd/quarrel

# -----------------------------------------------------------------------------
# Build Stage: macOS Universal Binary (arm64 + x86_64)
# -----------------------------------------------------------------------------
FROM --platform=linux/amd64 golang:1.27.0-alpine AS universal-builder

RUN apk add --no-cache build-base git

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build for both arm64 and x86_64, then combine into universal binary
RUN CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 CC=o64-clang go build -o quarrel-metal-arm64 ./cmd/quarrel && \
    CGO_ENABLED=1 GOOS=darwin GOARCH=amd64 CC=o64-clang go build -o quarrel-metal-amd64 ./cmd/quarrel && \
    lipo -create -arch arm64 quarrel-metal-arm64 -arch x86_64 quarrel-metal-amd64 -output quarrel-metal

# -----------------------------------------------------------------------------
# Runtime Stage: Minimal Runtime
# -----------------------------------------------------------------------------
FROM alpine:latest

WORKDIR /app

# Copy binary
COPY --from=metal-builder /app/quarrel-metal ./quarrel

CMD ["./quarrel", "--help"]

# =============================================================================
# Native macOS Build Instructions
# =============================================================================
# On macOS with Metal installed:
#
#   CGO_ENABLED=1 go build -o quarrel-metal ./cmd/quarrel
#   ./quarrel --model model.gguf
#
# Or build universal binary:
#
#   CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 go build -o quarrel-arm64 ./cmd/quarrel
#   CGO_ENABLED=1 GOOS=darwin GOARCH=amd64 go build -o quarrel-amd64 ./cmd/quarrel
#   lipo -create -arch arm64 quarrel-arm64 -arch x86_64 quarrel-amd64 -output quarrel
#
# =============================================================================
# GPU Requirements
# =============================================================================
# - macOS 12.0+ (Monterey or later)
# - Apple Silicon (M1/M2/M3) or Intel with Metal-capable GPU
# - Metal is enabled by default on macOS
#
# Run:
#   docker run -v $(pwd)/models:/data longbow-quarrel:metal --model /data/model.gguf
#
# =============================================================================