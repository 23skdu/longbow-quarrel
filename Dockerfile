# Build stage
FROM golang:1.23-alpine AS builder

# Install build dependencies
RUN apk add --no-cache build-base

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
# Note: Metal support is macOS-only. This build will not include Metal acceleration.
RUN CGO_ENABLED=0 GOOS=linux go build -o quarrel ./cmd/quarrel

# Final stage
FROM alpine:latest

WORKDIR /app

# Copy the binary from the builder stage
COPY --from=builder /app/quarrel .

# Set entrypoint
ENTRYPOINT ["./quarrel"]
