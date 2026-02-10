# Longbow-Quarrel WebUI Usage Guide

## Overview

The Longbow-Quarrel WebUI provides a responsive web interface for interacting with the LLM inference engine via WebSockets and REST APIs.

## Quick Start

### Local Development

```bash
# Build and run with CUDA support
cd cmd/webui
go build -tags "webui,linux,cuda" -o webui .
./webui --port 8080

# Or with CPU-only
go build -tags "webui" -o webui .
./webui --port 8080
```

### Docker

```bash
# GPU-enabled version
docker compose -f docker-compose.webui.yml --profile cuda up -d

# CPU-only version
docker compose -f docker-compose.webui.yml --profile cpu up -d

# With monitoring stack
docker compose -f docker-compose.webui.yml --profile cuda --profile monitoring up -d
```

### Access WebUI

Open http://localhost:8080 in your browser.

---

## Environment Variables

### WebUI Configuration

| Variable | Flag | Default | Description |
|----------|------|---------|-------------|
| `WEBUI_PORT` | `--port` | 8080 | HTTP server port |
| `WEBUI_METRICS_PORT` | `--metrics-port` | 9090 | Prometheus metrics port |
| `WEBUI_HOST` | `--host` | 0.0.0.0 | Host to bind to |
| `WEBUI_DATA_DIR` | N/A | /data | Directory for model files |
| `WEBUI_CACHE_DIR` | N/A | /data/cache | Directory for KV cache |

### Authentication

| Variable | Flag | Default | Description |
|----------|------|---------|-------------|
| `WEBUI_API_KEY` | `--api-key` | "" | API key for authentication (optional) |
| `WEBUI_ALLOWED_ORIGINS` | `--allowed-origins` | "*" | Comma-separated CORS origins |

### Examples

```bash
# With API key authentication
export WEBUI_API_KEY="qk_your_secure_api_key_here"
./webui --api-key "$WEBUI_API_KEY"

# With custom CORS origins
export WEBUI_ALLOWED_ORIGINS="http://localhost:3000,https://app.example.com"
./webui --allowed-origins "$WEBUI_ALLOWED_ORIGINS"

# All options combined
./webui \
  --port 8080 \
  --metrics-port 9090 \
  --host 0.0.0.0 \
  --api-key "qk_your_key" \
  --allowed-origins "https://yourdomain.com"
```

---

## Command Line Flags

```bash
./webui [options]

Options:
  --port           int     HTTP server port (default 8080)
  --metrics-port   int     Prometheus metrics port (default 9090)
  --host          string  Host to bind to (default "0.0.0.0")
  --api-key       string  API key for authentication
  --allowed-origins string  Comma-separated CORS origins
```

---

## Docker Deployment

### Docker Compose Profiles

```bash
# GPU + Monitoring
docker compose -f docker-compose.webui.yml --profile cuda --profile monitoring up -d

# CPU-only
docker compose -f docker-compose.webui.yml --profile cpu up -d

# With Nginx reverse proxy
docker compose -f docker-compose.webui.yml --profile cuda --profile nginx up -d

# All services
docker compose -f docker-compose.webui.yml --profile cuda --profile monitoring --profile nginx up -d
```

### Environment File

Create a `.env` file:

```bash
# .env
WEBUI_API_KEY=qk_your_secure_api_key
WEBUI_ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=secure_password
```

### Docker Run Command

```bash
docker run -d \
  --name longbow-quarrel-webui \
  -p 8080:8080 \
  -p 9090:9090 \
  -v $(pwd)/models:/data/models:ro \
  -v $(pwd)/cache:/data/cache \
  -e WEBUI_API_KEY=your_api_key \
  -e WEBUI_ALLOWED_ORIGINS="https://yourdomain.com" \
  longbow-quarrel-webui:latest
```

---

## API Usage

### WebSocket API

Connect to `ws://localhost:8080/ws` for bidirectional communication.

#### Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
    console.log('Connected');
};

ws.onerror = (error) => {
    console.error('Error:', error);
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
};
```

#### Send Inference Request

```javascript
ws.send(JSON.stringify({
    type: 'inference',
    payload: {
        prompt: 'Explain quantum computing',
        temperature: 0.7,
        top_k: 40,
        top_p: 0.95,
        max_tokens: 512,
        stream: true
    }
}));
```

#### Receive Responses

```javascript
ws.onmessage = (event) => {
    const { type, payload } = JSON.parse(event.data);

    switch (type) {
        case 'inference':
            console.log('Token:', payload.token);
            console.log('Complete:', payload.complete);
            console.log('Tokens/sec:', payload.tokens_per_sec);
            break;
        case 'status':
            console.log('State:', payload.state);
            break;
        case 'error':
            console.error('Error:', payload.message);
            break;
    }
};
```

#### Stop Generation

```javascript
ws.send(JSON.stringify({
    type: 'stop',
    payload: {}
}));
```

### REST API

All REST endpoints require authentication if `WEBUI_API_KEY` is set.

#### List Models

```bash
curl -H "Authorization: ApiKey YOUR_API_KEY" \
  http://localhost:8080/api/models
```

Response:
```json
[
  {
    "name": "smollm2",
    "path": "/models/smollm2.gguf",
    "loaded": true,
    "parameters": "1.7B",
    "quantization": "Q4_K_M"
  }
]
```

#### Generate Text (Single-shot)

```bash
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: ApiKey YOUR_API_KEY" \
  -d '{
    "prompt": "Explain quantum computing in simple terms.",
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95,
    "max_tokens": 100
  }'
```

Response:
```json
{
  "text": "Quantum computing uses quantum bits (qubits)...",
  "tokens_generated": 42,
  "tokens_per_sec": 125.5,
  "finish_reason": "stop"
}
```

#### Generate Text (Streaming)

```bash
curl -X POST http://localhost:8080/api/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: ApiKey YOUR_API_KEY" \
  -d '{
    "prompt": "Write a Python hello world.",
    "max_tokens": 50,
    "stream": true
  }'
```

Server-Sent Events response:
```
data: {"token":"def","token_id":0,"complete":false,"tokens_per_sec":0}
data: {"token":" hello","token_id":1,"complete":false,"tokens_per_sec":50.2}
data: {"token":" world","token_id":2,"complete":false,"tokens_per_sec":75.3}
...
data: {"token":"\n","token_id":49,"complete":true,"tokens_per_sec":98.5}
```

---

## Health Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Detailed health status (JSON) |
| `/healthz` | Simple liveness probe |
| `/readyz` | Readiness probe |
| `/version` | Version information |

```bash
# Detailed health
curl http://localhost:8080/health

# Response
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "0.0.1",
  "uptime": "2h30m45s",
  "checks": {
    "server": { "status": "healthy" },
    "memory": { "status": "healthy" },
    "goroutines": { "status": "healthy" }
  }
}
```

---

## Metrics

Prometheus metrics available at `http://localhost:9090/metrics`.

See [Metrics Documentation](metrics.md) for detailed metrics reference.

---

## WebUI Features

### Theme Toggle

Click the sun/moon icon in the header to toggle between dark and light themes. Theme preference is saved in localStorage.

### Model Selection

Select from available models in the sidebar dropdown. Model information (parameters, quantization) is displayed when selected.

### Settings

Adjust sampling parameters:
- **Temperature** (0.0-2.0): Higher = more creative, lower = more focused
- **Top K** (1-100): Number of top tokens to consider
- **Top P** (0.0-1.0): Nucleus sampling threshold
- **Max Tokens**: Maximum tokens to generate

### Conversation

- **New Chat**: Start a new conversation (clears current)
- **Clear History**: Remove all stored conversations
- **Export**: Download conversation as Markdown

### Quick Prompts

Click quick prompt buttons to populate the input:
- **Summarize**: "Summarize this text"
- **Explain**: "Explain this concept"
- **Translate**: "Translate to Spanish"
- **Code**: "Write code for"

---

## Security

### API Key Management

Generate a secure API key:

```bash
# Generate using OpenSSL
openssl rand -hex 32

# Or use the built-in generator
go run -tags webui ./cmd/webui -generate-key
```

### Rate Limiting

Default rate limit: 100 requests per minute per API key.

### CORS Configuration

For production, specify allowed origins:

```bash
./webui --allowed-origins "https://yourdomain.com,https://app.yourdomain.com"
```

Avoid wildcard (`*`) in production except for development.

---

## Troubleshooting

### GPU Not Detected

```bash
# Verify CUDA installation
nvidia-smi

# Check container GPU access
docker run --rm --gpus all nvidia/cuda:12.2-base-ubuntu22.04 nvidia-smi
```

### WebUI Not Starting

Check logs:
```bash
# Docker
docker logs longbow-quarrel-webui

# Local
./webui --port 8080 2>&1
```

### Connection Refused

Verify the port is not in use:
```bash
lsof -i :8080
```

### Memory Issues

```bash
# Check available memory
free -h

# Container memory limits
docker stats
```

---

## Performance Tuning

### Recommended Settings

| Hardware | Max Concurrent | Memory per Request |
|----------|---------------|-------------------|
| RTX 3090 (24GB) | 4 | 4GB |
| A100 (40GB) | 8 | 4GB |
| CPU (32GB) | 2 | 8GB |

### Model Quantization

Use Q4_K_M or Q5_K_M quantization for best performance/quality ratio.

---

## Command Line Options (Legacy CLI)

### General

| Flag | Default | Description |
|------|---------|-------------|
| `-model` | (required) | Path to GGUF model file, or Ollama model name |
| `-prompt` | "Hello world" | Input prompt text |
| `-n` | 20 | Number of tokens to generate |
| `-stream` | false | Stream tokens to stdout as generated |
| `-metrics` | :9090 | Address for Prometheus metrics |
| `-gpu` | false | Explicitly request Metal GPU acceleration |

### Sampling & Quality

| Flag | Default | Description |
|------|---------|-------------|
| `-temp` | 0.7 | Sampling Temperature (0.0 = greedy) |
| `-topk` | 40 | Top-K Sampling |
| `-topp` | 0.95 | Top-P (Nucleus) Sampling |
| `-penalty` | 1.1 | Repetition Penalty (1.0 = none) |
| `-quality` | false | Enable advanced quality-guided sampling |
| `-chatml` | false | Wrap prompt in ChatML template |

### Advanced / Debugging

| Flag | Default | Description |
|------|---------|-------------|
| `-kv-cache-size` | 22 | KV cache size in MiB |
| `-benchmark` | false | Run in benchmark mode |
| `-debug-dequant` | false | Enable dequantization debug dump |
| `-debug-activations` | false | Enable layer-by-layer activation dumping |
| `-debug-perf` | false | Enable performance metric logging for kernels |

---

## Model Format

Longbow-quarrel supports GGUF format models with Llama 3 architecture:

```bash
# Example: Download smollm2 via Ollama
ollama pull smollm2:135m-instruct
```

Supported models:
- smollm2-135M/360M-instruct
- Mistral-7B-v0.1/v0.3
- Llama 3.2-1B/3B
- Any Llama 3/Mistral architecture GGUF model

### Supported Quantizations

- **K-Quantization**: Q3_K, Q4_K, Q6_K (fully accelerated)
- **Standard**: FP16, FP32

---

## Further Reading

- [API Reference](metrics.md) - Prometheus metrics
- [Architecture](architecture.md)
- [Development Guide](development.md)
