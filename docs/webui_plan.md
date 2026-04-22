# Templ WebUI Service Plan

## Overview

Add a responsive web-based UI for interacting with the Quarrel inference engine via WebSockets, enabling service access to the LLM from any browser or client.

---

## Part 1: Project Setup & Dependencies

**Objective:** Initialize Templ project structure and dependencies

### 1.1 Initialize Go Module
- [ ] Create `cmd/webui/` directory structure
- [ ] Initialize Go module with required dependencies:
  - `github.com/a-h/templ` - Templ templating engine
  - `github.com/gorilla/websocket` - WebSocket support
  - `github.com/gin-gonic/gin` - HTTP routing (optional, for REST endpoints)
  - `github.com/prometheus/client_golang` - Metrics endpoint

### 1.2 Project Structure
```
cmd/webui/
├── main.go                 # Entry point
├── handlers/
│   ├── websocket.go       # WebSocket handler
│   ├── inference.go       # Inference API handler
│   └── metrics.go        # Prometheus metrics
├── templates/
│   ├── base.templ        # Base template
│   ├── index.templ       # Chat interface
│   └── components/
│       ├── chat.templ    # Chat message component
│       ├── sidebar.templ # Model selection sidebar
│       └── settings.templ # Settings panel
├── static/
│   ├── css/              # Stylesheets
│   └── js/               # Client-side JavaScript
└── config/
    └── config.go         # WebUI configuration
```

### 1.3 Build Configuration
- [ ] Add build tag `//go:build webui` to webui-specific files
- [ ] Update `Dockerfile` with multi-stage build for webui
- [ ] Add `docker-compose.webui.yml` for local development

**Deliverable:** Project structure ready for component development

---

## Part 2: WebSocket Infrastructure

**Objective:** Implement bidirectional communication between browser and inference engine

### 2.1 WebSocket Handler
- [ ] Create `cmd/webui/handlers/websocket.go`
- [ ] Implement `HandleWebSocket()`:
  - Upgrade HTTP to WebSocket connection
  - Manage connection lifecycle (connect/disconnect)
  - Handle ping/pong for connection health
  - Implement reconnection logic

### 2.2 Message Protocol
Define JSON message format:
```go
type WSMessage struct {
    Type    string      `json:"type"`    // "inference", "status", "error", "metrics"
    Payload interface{} `json:"payload"`
}

type InferenceRequest struct {
    Prompt      string            `json:"prompt"`
    Model       string            `json:"model,omitempty"`
    Temperature float64           `json:"temperature,omitempty"`
    TopK        int               `json:"topk,omitempty"`
    TopP        float64           `json:"topp,omitempty"`
    MaxTokens   int               `json:"max_tokens,omitempty"`
    Stream      bool              `json:"stream"`
}

type InferenceResponse struct {
    Token       string            `json:"token"`
    TokenID     int               `json:"token_id"`
    Stream      bool              `json:"stream"`
    Complete    bool              `json:"complete"`
    TokensPerSec float64          `json:"tokens_per_sec"`
}
```

### 2.3 Connection Manager
- [ ] Implement `ConnectionManager`:
  - Track active connections by model
  - Broadcast messages to connected clients
  - Rate limiting per connection
  - Max concurrent connections per model

### 2.4 Error Handling
- [ ] Define error codes:
  - `INVALID_REQUEST`, `MODEL_NOT_FOUND`, `INFERENCE_ERROR`, `CONNECTION_LOST`
- [ ] Implement graceful degradation
- [ ] Log errors with request IDs for debugging

**Deliverable:** Functional WebSocket infrastructure with JSON protocol

---

## Part 3: Inference Engine Integration

**Objective:** Connect WebSocket handler to existing Quarrel inference engine

### 3.1 Engine Adapter
- [ ] Create `cmd/webui/engine/adapter.go`
- [ ] Implement `InferenceAdapter`:
  - Wrap `engine.NewEngine()` for concurrent requests
  - Manage engine lifecycle (load/unload models)
  - Handle model hot-swapping
  - Implement model caching

### 3.2 Streaming Response
- [ ] Implement streaming token generation:
  - Yield tokens as they're generated
  - Track tokens-per-second in real-time
  - Support early termination (stop generation)

### 3.3 Request Queue
- [ ] Implement request prioritization:
  - High priority: Interactive requests
  - Low priority: Batch/generation requests
- [ ] Add backpressure handling when queue full
- [ ] Implement request timeout (default 5 minutes)

### 3.4 KV Cache Sharing
- [ ] Support KV cache persistence between requests:
  - Cache prefix prompts for faster completion
  - Implement cache invalidation
  - Add cache size limits per model

**Deliverable:** Integration layer connecting WebSocket to inference engine

---

## Part 4: Base UI Components (Templ)

**Objective:** Create reusable Templ components for the web interface

### 4.1 Base Template
- [ ] Create `cmd/webui/templates/base.templ`
- [ ] Include:
  - HTML5 doctype
  - Meta tags for responsiveness
  - CSP headers for security
  - Preload critical assets

### 4.2 Layout Components
- [ ] Create layout components in `templates/components/`:
  - `header.templ` - Title, connection status
  - `footer.templ` - Version, links
  - `container.templ` - Main content wrapper

### 4.3 Chat Interface
- [ ] Create `cmd/webui/templates/components/chat.templ`:
  - Message list (user/assistant distinction)
  - Streaming token display
  - Typing indicator
  - Auto-scroll on new messages

### 4.4 Responsive Design
- [ ] Implement mobile-first CSS:
  - Sidebar collapses on mobile
  - Touch-friendly message input
  - Portrait/landscape optimizations
  - Dark/light theme toggle

**Deliverable:** Responsive base UI components in Templ

---

## Part 5: Interactive Features

**Objective:** Add interactive elements for model control

### 5.1 Model Selection Sidebar
- [ ] Create `cmd/webui/templates/components/sidebar.templ`:
  - List available models
  - Show model status (loaded/unloaded)
  - Model info (parameters, quantization)
  - Memory usage indicator

### 5.2 Settings Panel
- [ ] Create `cmd/webui/templates/components/settings.templ`:
  - Temperature slider (0.0-2.0)
  - TopK input (1-100)
  - TopP slider (0.0-1.0)
  - Max tokens input
  - Reset to defaults button

### 5.3 Conversation History
- [ ] Implement conversation management:
  - Save conversation to localStorage
  - Load previous conversations
  - Clear history
  - Export conversation as JSON/Markdown

### 5.4 Prompt Templates
- [ ] Add preset prompts:
  - "Summarize", "Translate", "Code", "Explain"
  - Custom prompt input
  - System prompt configuration

**Deliverable:** Full-featured interactive sidebar and settings panel

---

## Part 6: Client-Side JavaScript

**Objective:** Implement WebSocket client and DOM manipulation

### 6.1 WebSocket Client
- [ ] Create `cmd/webui/static/js/websocket.js`:
  - Connection establishment
  - Message serialization/deserialization
  - Reconnection with exponential backoff
  - Heartbeat mechanism

### 6.2 UI State Management
- [ ] Implement state machine:
  - States: `disconnected`, `connecting`, `connected`, `generating`
  - Visual feedback for each state
  - Disable inputs during generation

### 6.3 DOM Updates
- [ ] Create `cmd/webui/static/js/ui.js`:
  - Efficient DOM updates (avoid reflows)
  - Virtual scrolling for long conversations
  - Markdown rendering (use `marked.js`)
  - Syntax highlighting for code blocks

### 6.4 Local Storage
- [ ] Implement persistence:
  - Save conversations
  - Persist settings
  - Store API keys (encrypted)
  - History search functionality

**Deliverable:** Client-side JavaScript for full interactivity

---

## Part 7: Styling (CSS)

**Objective:** Create modern, responsive styles

### 7.1 CSS Architecture
- [ ] Create `cmd/webui/static/css/main.css`:
  - CSS variables for theming
  - Mobile-first breakpoints
  - BEM naming convention
  - Minified production build

### 7.2 Theme Support
- [ ] Implement dark/light themes:
  - System preference detection
  - Manual toggle
  - Smooth transitions
  - Consistent color palette

### 7.3 Component Styles
- [ ] Style key components:
  - Chat bubbles (user: blue, assistant: gray)
  - Sidebar (collapsible on mobile)
  - Settings panel (modal/slide-out)
  - Loading indicators (spinners, typing animation)

### 7.4 Animations
- [ ] Add micro-interactions:
  - Message fade-in
  - Typing indicator
  - Button hover effects
  - Connection status pulse

**Deliverable:** Complete CSS styling with dark mode support

---

## Part 8: REST API Endpoints

**Objective:** Provide REST endpoints for non-WebSocket clients

### 8.1 API Routes
Implement REST endpoints in `cmd/webui/handlers/`:
```
GET  /api/models              - List available models
GET  /api/models/:name        - Get model info
POST /api/generate            - Single-shot generation
POST /api/stream              - Streaming generation (SSE)
GET  /api/health              - Health check
GET  /api/metrics             - Prometheus metrics
```

### 8.2 Authentication
- [ ] Add API key authentication:
  - Generate API keys via CLI
  - Validate keys on each request
  - Rate limiting per key
  - Key rotation support

### 8.3 OpenAPI Spec
- [ ] Create `cmd/webui/api/openapi.yaml`:
  - Document all endpoints
  - Generate client SDKs
  - Interactive API documentation

### 8.4 CORS Support
- [ ] Configure CORS for cross-origin requests
- [ ] Support preflight OPTIONS requests
- [ ] Configurable allowed origins

**Deliverable:** REST API with OpenAPI documentation

---

## Part 9: Metrics & Observability

**Objective:** Add comprehensive monitoring

### 9.1 Prometheus Metrics
Create metrics in `cmd/webui/handlers/metrics.go`:
- `quarrel_webui_connections_active` - Active WebSocket connections
- `quarrel_webui_requests_total` - Total requests by model
- `quarrel_webui_inference_duration_seconds` - Inference latency
- `quarrel_webui_tokens_total` - Total tokens generated
- `quarrel_webui_errors_total` - Error count by type

### 9.2 Structured Logging
- [ ] Implement structured logging:
  - JSON format for log aggregation
  - Request IDs for tracing
  - Log levels (DEBUG, INFO, WARN, ERROR)
  - Sensitive data redaction

### 9.3 Health Endpoints
- [ ] Add health checks:
  - `/healthz` - Liveness probe
  - `/readyz` - Readiness probe (checks engine status)
  - `/version` - Version info

### 9.4 Tracing
- [ ] Add distributed tracing:
  - Trace inference requests
  - Span for WebSocket message processing
  - Export to Jaeger/Zipkin (optional)

**Deliverable:** Complete observability stack

---

## Part 10: Deployment & Testing

**Objective:** Production-ready deployment and test coverage

### 10.1 Docker Configuration
- [ ] Create `Dockerfile.webui`:
  - Multi-stage build
  - Non-root user
  - Health checks
  - Resource limits

### 10.2 Docker Compose
- [ ] Create `docker-compose.webui.yml`:
  - WebUI service
  - Prometheus + Grafana dashboard
  - Optional: Nginx reverse proxy

### 10.3 Unit Tests
Create test files:
- [ ] `handlers/websocket_test.go` - WebSocket handler tests
- [ ] `handlers/inference_test.go` - API handler tests
- [ ] `engine/adapter_test.go` - Engine adapter tests
- [ ] `templates/components_test.go` - Template rendering tests

### 10.4 Integration Tests
- [ ] Create `cmd/webui/test/e2e/`:
  - WebSocket full-duplex test
  - Multi-client concurrent connections
  - Model hot-swap test
  - Failure recovery test
  - Load test (100+ concurrent connections)

### 10.5 Load Testing
- [ ] Benchmark script:
  - Measure throughput (tokens/second)
  - Latency percentiles (p50, p95, p99)
  - Memory footprint
  - Connection scalability

**Deliverable:** Production deployment with comprehensive test coverage

---

## Implementation Order

| Phase | Focus | Duration |
|-------|-------|----------|
| 1 | Project setup & WebSocket | Week 1 |
| 2 | Engine integration & REST API | Week 1-2 |
| 3 | Templ components & styling | Week 2-3 |
| 4 | Client JavaScript & interactivity | Week 3 |
| 5 | Metrics, deployment & testing | Week 4 |

**Target Completion:** 4-5 weeks for production-ready WebUI service

---

## Quick Start

```bash
# Run webui locally
go run -tags webui ./cmd/webui/

# With custom port
WEBUI_PORT=8080 go run -tags webui ./cmd/webui/

# Docker
docker build -f Dockerfile.webui -t quarrel-webui .
docker run -p 8080:8080 quarrel-webui

# Open browser
# http://localhost:8080
```

---

## API Usage Examples

### WebSocket (JavaScript)
```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'inference',
        payload: { prompt: 'Hello,', stream: true }
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'inference') {
        console.log(data.payload.token);
    }
};
```

### REST API (cURL)
```bash
# Generate response
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "max_tokens": 100}'
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `github.com/a-h/templ` | HTML templating |
| `github.com/gorilla/websocket` | WebSocket handling |
| `github.com/prometheus/client_golang` | Metrics |
| `github.com/gin-gonic/gin` | HTTP routing |
| `github.com/markbates/grift` | CLI tools |
