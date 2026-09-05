# Longbow-Quarrel API Reference

## HTTP Endpoints

### OpenAI-Compatible API

#### POST /v1/chat/completions

Chat completion endpoint.

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

**Request Body:**
```json
{
  "model": "string",
  "messages": [{"role": "user|assistant", "content": "string"}],
  "temperature": 0.7,
  "top_p": 0.95,
  "max_tokens": 256,
  "stream": false
}
```

> **Note on `model` parameter:** Supports fuzzy model names (e.g. `"Qwen3.5"`, `"mistral:latest"`, `"Llama-3.2-3B"`) which automatically resolve across `~/.cache/llmfit/models/`, `~/.cache/llama.cpp/`, `~/.cache/huggingface/hub/`, and `~/.ollama/models/`, in addition to exact paths.


**Response:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "default",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

#### POST /v1/completions

Legacy completion endpoint.

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "Hello",
    "max_tokens": 100
  }'
```

**Extensions (Phase 3):**
- `"adapter"`: (string) ID of a previously loaded LoRA adapter.
- `"speculative"`: (bool) Enable accelerated speculative decoding.
- `"draft_k"`: (int, optional) Number of tokens to draft per step (default: 4).
- `"num_paths"`: (int, optional) Number of parallel sampling paths (default: 1).

### Custom Endpoints

#### POST /generate

Simple generation without OpenAI format.

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Your prompt",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

#### POST /stream

Streaming generation via Server-Sent Events.

```bash
curl -X POST http://localhost:8080/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Your prompt",
    "max_tokens": 100
  }'
```

#### WebSocket /ws

WebSocket streaming for real-time token generation.

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');
ws.onmessage = (event) => {
  console.log(JSON.parse(event.data));
};
```

### Model Management

#### GET /models

List available models.

```bash
curl http://localhost:8080/models
```

#### POST /hotswap

Hot-swap model without restart.

```bash
curl -X POST http://localhost:8080/hotswap \
  -H "Content-Type: application/json" \
  -d '{"model": "/path/to/new_model.gguf"}'
```

#### POST /v1/adapters/load
Load a LoRA adapter dynamically.

```bash
curl -X POST http://localhost:8080/v1/adapters/load \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/path/to/adapter.gguf",
    "id": "my-lora-id"
  }'
```

#### GET /v1/adapters/list
List all currently loaded LoRA adapters.

```bash
curl http://localhost:8080/v1/adapters/list
```

### Health & Metrics

| Endpoint | Description |
|----------|-------------|
| GET /health | Health check |
| GET /healthz | Minimal health |
| GET /readyz | Readiness check |
| GET /version | Version info |
| GET /metrics | Prometheus metrics |
| GET / | WebUI |

### Authentication

All endpoints (except /health, /healthz, /metrics) require API key:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" ...
```

Default API key configured via `QUARREL_API_KEY` environment variable.

## Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 500 | Server Error |