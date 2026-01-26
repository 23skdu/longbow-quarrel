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

# Build the application for Linux (no Metal support)
RUN CGO_ENABLED=0 GOOS=linux go build -o quarrel-linux ./cmd/quarrel

# Build stage for macOS (with Metal/CGO support)
FROM --platform=linux/amd64 golang:1.23-alpine AS darwin-builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application for macOS (with Metal/CGO support)
RUN CGO_ENABLED=1 GOOS=darwin CC=o64-clang go build -o quarrel-darwin ./cmd/quarrel

# Final stage
FROM alpine:latest

WORKDIR /app

# Copy Linux binary (primary target for containerized deployment)
COPY --from=linux-builder /app/quarrel-linux ./quarrel

# Set entrypoint
ENTRYPOINT ["./quarrel"]

# Note: For macOS development, use quarrel-darwin binary with Metal acceleration
# The darwin-builder stage creates quarrel-darwin for local macOS development
