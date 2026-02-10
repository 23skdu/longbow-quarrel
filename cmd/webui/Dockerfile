# Longbow-Quarrel WebUI Docker Image

FROM golang:1.25-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -tags webui -ldflags="-s -w" -o webui ./cmd/webui/

FROM alpine:3.19

RUN apk --no-cache add ca-certificates

WORKDIR /app

COPY --from=builder /app/webui .
COPY --from=builder /app/cmd/webui/static ./static
COPY --from=builder /app/cmd/webui/templates ./templates

EXPOSE 8080 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

ENV PORT=8080
ENV METRICS_PORT=9090

CMD ["sh", "-c", "./webui --port ${PORT} --metrics-port ${METRICS_PORT}"]
