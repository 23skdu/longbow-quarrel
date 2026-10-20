# Longbow-Quarrel API Reference

> **Scope.** This document describes release **v0.3.0**'s API. `main`
> carries additional unreleased work; see [nextsteps.md](nextsteps.md) for
> what has landed and what is planned.

The server listens on **two** addresses, and they are separate listeners:

| Listener | Flag | Default | Serves |
|----------|------|---------|---------|
| The API | `-api-addr` | `:8080` | The completions, the adapters and the health probes |
| The metrics | `-metrics` | `:9090` | `/metrics`, and nothing else |

The API server does **not** serve `/metrics`, and the metrics server does
**not** serve the completions. A deployment that scrapes the API port gets
nothing.

The WebUI is a separate binary (`cmd/webui`) with its own routes; see
[webui_plan.md](webui_plan.md).

## The API listener

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

**Chat Template Rendering:**
Quarrel automatically applies the `tokenizer.chat_template` Jinja2 string from GGUF KV metadata when processing chat messages. This ensures correct prompt formatting for Llama 3, Qwen, Gemma, Mistral, and other instruction-tuned models. No additional configuration is needed — the template is detected and applied server-side.

**Extensions (Implemented):**
- `"adapter"`: (string) ID of a previously loaded LoRA adapter.
- `"speculative"`: (bool) Enable speculative decoding with multi-token draft verification. The server has to be started with a draft model (`quarrel -draft-model <path>`); without one the field is ignored and the request is served by the plain engine.
- `"draft_k"`: (int, optional) Number of tokens to draft per step (default: 4).

### Adapters

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

### Health

| Endpoint | Description |
|----------|-------------|
| GET /healthz | Liveness: the process is up |
| GET /readyz | Readiness: an engine is loaded and can serve |

Both are on the API listener, so a Kubernetes probe can use either.

## The metrics listener

| Endpoint | Description |
|----------|-------------|
| GET /metrics | The Prometheus metrics |

See [metrics.md](metrics.md) for the full list.

## The endpoints that do not exist

Earlier revisions of this document listed a set of endpoints that the
server never registered: `/generate`, `/stream`, `/ws`, `/models`,
`/hotswap`, `/health`, `/version` and `/`. They are gone, and they are
listed here so that nobody tries them again:

| Endpoint | Where it actually is |
|----------|----------------------|
| `POST /generate`, `POST /stream` | Nowhere. Use `/v1/completions` and `/v1/chat/completions`. |
| `WebSocket /ws` | The WebUI binary (`cmd/webui`), on its own port. |
| `GET /models`, `POST /hotswap` | Nowhere. The model is chosen at start-up. |
| `GET /health`, `GET /version` | The WebUI binary. The API server has `/healthz` and `/readyz`. |
| `GET /` | The WebUI binary. |
| `GET /metrics` | The metrics listener, not the API one. |

## Authentication

The API listener has **no authentication**. It serves the completions, the
adapter endpoints and the health probes to anything that can reach it, so
it belongs on a private network or behind a gateway that does the
authentication.

The WebUI binary has an API key, configured with the `-api-key` flag or
the `WEBUI_API_KEY` environment variable.

> The variable is `WEBUI_API_KEY`, not `QUARREL_API_KEY`: earlier revisions
> of this document named a variable that does not exist, so the API key was
> never actually applied.

## Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 404 | Not Found |
| 500 | Server Error |
