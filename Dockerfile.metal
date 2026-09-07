# =============================================================================
# Longbow-Quarrel - Apple Metal Build
# =============================================================================
# This Dockerfile builds Quarrel with Metal acceleration for macOS arm64.
# Uses Apple Metal for GPU acceleration on Apple Silicon.
#
# Note: This build is designed to run on macOS (darwin/arm64).
# Linux containers CANNOT execute darwin binaries. This Dockerfile
# cross-compiles the binary; the runtime stage below is only a
# carrier for artifact extraction (use `docker build --output` or
# `docker cp`). For production use, build natively on macOS:
#   CGO_ENABLED=1 go build -o quarrel-metal ./cmd/quarrel
#
# Cross-compiling from Linux additionally requires osxcross
# (o64-clang) + cctools (lipo) + macOS SDK, which are NOT installed
# by default. Install them in your own builder image, or build on Mac.
#
# Build:    docker build -f Dockerfile.metal -t longbow-quarrel:metal .
# Run:      docker run --device metal -v $(pwd)/models:/data longbow-quarrel:metal --model /data/model.gguf
# =============================================================================

# -----------------------------------------------------------------------------
# Build Stage: Metal Compilation (cross-compile from Linux amd64 to darwin arm64)
# Requires osxcross (o64-clang) + macOS SDK - see note above.
# -----------------------------------------------------------------------------
FROM --platform=linux/amd64 golang:1.27.0-alpine AS metal-builder

# Install build dependencies for cross-compilation
# NOTE: o64-clang/lipo come from osxcross/cctools, not stock alpine.
# Uncomment and point at your internal osxcross image to enable:
#   COPY --from=osxcross-image /osxcross /osxcross
#   ENV PATH=/osxcross/bin:$PATH
RUN apk add --no-cache build-base git clang llvm
RUN echo "WARNING: o64-clang (osxcross) + macOS SDK required for darwin cross-compile - see Dockerfile header" >&2

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build with CGO for Metal (darwin/arm64)
# Requires CC=o64-clang from osxcross; plain clang cannot target darwin.
RUN if ! command -v o64-clang >/dev/null 2>&1; then echo "ERROR: o64-clang not found. Install osxcross or build natively on macOS (see header)." >&2; exit 1; fi && \
    CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 CC=o64-clang go build -o quarrel-metal ./cmd/quarrel

# -----------------------------------------------------------------------------
# Runtime Stage: Artifact carrier (darwin binary cannot execute on Linux)
# Extract with: docker create --name m longbow-quarrel:metal && docker cp m:/app/quarrel ./quarrel-metal
# -----------------------------------------------------------------------------
FROM alpine:3.19

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