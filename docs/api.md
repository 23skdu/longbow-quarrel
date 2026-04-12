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