# Longbow-Quarrel API Client

A Python client library for interacting with the Longbow-Quarrel WebUI API. Supports CRUD operations, API validation, and administrative functions.

## Installation

No external dependencies required - uses only Python standard library.

```bash
# Ensure the script is executable
chmod +x scripts/api_client.py
```

## Quick Start

```bash
# Check health
python scripts/api_client.py health

# List models
python scripts/api_client.py models

# Generate text
python scripts/api_client.py generate -p "Hello, how are you?"

# Run full validation
python scripts/api_client.py validate
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LONGBOW_URL` | API base URL | `http://localhost:8080` |
| `LONGBOW_API_KEY` | API key for authentication | (none) |

### Command-line Options

```
--url URL           Base URL of the API (default: http://localhost:8080)
--api-key KEY      API key for authentication
--timeout SECONDS  Request timeout (default: 60)
--verbose, -v      Verbose output
```

## Commands

### Health Checks

```bash
# Full health check with system status
python scripts/api_client.py health

# Simple liveness probe
python scripts/api_client.py healthz

# Readiness probe with memory/goroutine checks
python scripts/api_client.py readyz

# API version information
python scripts/api_client.py version

# Prometheus metrics
python scripts/api_client.py metrics
```

### Model Operations

```bash
# List available models
python scripts/api_client.py models
```

### Text Generation

```bash
# Synchronous generation
python scripts/api_client.py generate -p "Your prompt here" -m model-name

# Streaming generation
python scripts/api_client.py stream -p "Your prompt here"
```

### Validation & Testing

```bash
# Run full API validation suite
python scripts/api_client.py validate --verbose

# Run administrative checks
python scripts/api_client.py admin

# Run performance benchmark
python scripts/api_client.py benchmark --requests 20

# Test WebSocket connection
python scripts/api_client.py websocket
```

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Full health check with status and checks |
| `/healthz` | GET | Simple liveness probe |
| `/readyz` | GET | Readiness probe with system checks |
| `/version` | GET | API version information |
| `/metrics` | GET | Prometheus metrics endpoint |
| `/api/models` | GET | List available models |
| `/api/generate` | POST | Synchronous text generation |
| `/api/stream` | POST | Streaming text generation (SSE) |
| `/ws` | WebSocket | Real-time inference |

## Generate Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--prompt` | `-p` | (required) | Input prompt text |
| `--model` | `-m` | `default` | Model name to use |
| `--temperature` | `-t` | `0.7` | Sampling temperature (0.0-2.0) |
| `--max-tokens` | `-n` | `256` | Maximum tokens to generate |
| `--top-k` | | `40` | Top-K sampling parameter |
| `--top-p` | | `0.95` | Top-P (nucleus) sampling parameter |

## Using as a Python Module

You can import the client and use it programmatically in your Python code:

```python
import sys
sys.path.insert(0, 'scripts')
from api_client import LongbowClient

# Create client instance
client = LongbowClient(
    base_url="http://localhost:8080",
    api_key="your-api-key",  # optional
    timeout=60
)

# Health check
result = client.health()
print(result)

# Generate text
result = client.generate(
    prompt="Hello, world!",
    model="default",
    temperature=0.7,
    max_tokens=100
)
print(result['data'])
```

## Programmatic Usage Examples

### Validation Suite

```python
from api_client import LongbowClient

client = LongbowClient("http://localhost:8080")
results = validate_api(client, verbose=True)

print(f"Passed: {results['passed']}/{results['total']}")
```

### Benchmark

```python
from api_client import LongbowClient

client = LongbowClient("http://localhost:8080")
stats = run_benchmark(
    client,
    prompt="Explain quantum computing",
    num_requests=50,
    verbose=True
)

print(f"Avg latency: {stats['avg_latency']:.3f}s")
print(f"Throughput: {stats['avg_tokens_per_sec']:.2f} tokens/sec")
```

## WebSocket Usage

The WebSocket endpoint (`/ws`) supports real-time streaming:

```javascript
// JavaScript example
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'inference',
        payload: {
            prompt: 'Hello',
            max_tokens: 50,
            stream: true
        }
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Token:', data.payload.token);
};
```

## Authentication

When API key authentication is enabled, include the key:

```bash
# Via environment variable
export LONGBOW_API_KEY="your-secret-key"
python scripts/api_client.py models

# Via command-line
python scripts/api_client.py models --api-key "your-secret-key"
```

## Error Handling

The client returns structured error responses:

```python
result = client.generate(prompt="test")
if 'success' in result and not result['success']:
    print(f"Error: {result['error']}")
else:
    print(f"Generated: {result['data']['text']}")
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | Error (invalid arguments, API error, etc.) |

## See Also

- [API Documentation](openapi.yaml)
- [Usage Guide](usage.md)
- [Production Integration](production_integration.md)