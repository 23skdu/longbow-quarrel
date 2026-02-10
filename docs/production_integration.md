# Production Integration Plan

**Objective:** Connect WebUI to real Quarrel inference engine and prepare for production deployment.

---

## Phase 1: Engine Integration

### 1.1 Connect Engine Adapter to Real Inference

**Current State:** `cmd/webui/engine/adapter.go` has a placeholder that logs requests but doesn't call the actual engine.

**Target State:** Full integration with `internal/engine/engine.go`

**Tasks:**
- [ ] Import actual engine packages (with proper build tags)
- [ ] Replace placeholder inference with `engine.NewEngine()` and `engine.Infer()`
- [ ] Add tokenizer integration via `tokenizer.New()`
- [ ] Handle model loading/unloading lifecycle
- [ ] Implement error propagation from engine to WebSocket

**Files to Modify:**
- `cmd/webui/engine/adapter.go`

### 1.2 Build Tag Integration

The WebUI needs to work with both Metal (darwin) and CUDA (linux) backends.

**Tasks:**
- [ ] Add `//go:build webui && (darwin || linux)` to handlers
- [ ] Create conditional imports based on platform
- [ ] Handle Metal device API vs CUDA device API differences
- [ ] Add conditional compilation for engine.NewEngine

**Files to Create:**
- `cmd/webui/engine/adapter_metal.go` (darwin)
- `cmd/webui/engine/adapter_cuda.go` (linux)

---

## Phase 2: Production Features

### 2.1 API Authentication

**Tasks:**
- [ ] Add API key generation via CLI
- [ ] Implement middleware for key validation
- [ ] Add rate limiting per key
- [ ] Store keys in secure config

### 2.2 CORS Configuration

**Tasks:**
- [ ] Configure allowed origins from environment
- [ ] Handle preflight OPTIONS requests
- [ ] Add CORS middleware to HTTP handlers

### 2.3 OpenAPI Documentation

**Tasks:**
- [ ] Create `cmd/webui/api/openapi.yaml`
- [ ] Document all REST endpoints
- [ ] Add Swagger UI static files
- [ ] Generate client SDKs (TypeScript, Python)

---

## Phase 3: Load Testing

### 3.1 Concurrent Connection Tests

**Target:** 100+ concurrent WebSocket connections

**Tasks:**
- [ ] Create load test script using `ghz` or similar
- [ ] Test WebSocket connection pooling
- [ ] Measure memory footprint under load
- [ ] Identify bottlenecks

### 3.2 Throughput Benchmarks

**Metrics to Measure:**
- Tokens/second per connection
- Total throughput (all connections)
- Latency percentiles (p50, p95, p99)
- GPU utilization (when applicable)

---

## Phase 4: Deployment

### 4.1 Kubernetes Support

**Tasks:**
- [ ] Create `k8s/` directory with manifests
- [ ] Add Deployment, Service, HorizontalPodAutoscaler
- [ ] Configure resource limits (CPU, memory, GPU)
- [ ] Add ConfigMap for configuration

### 4.2 Helm Chart

**Tasks:**
- [ ] Create `helm/quarrel-webui/` chart
- [ ] Support values for replica count, resources, ingress
- [ ] Add probes (liveness, readiness)
- [ ] Configure autoscaling

---

## Implementation Priority

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | Connect engine adapter | 2 days | Critical |
| 2 | Build tag integration | 1 day | Critical |
| 3 | API authentication | 1 day | Security |
| 4 | CORS configuration | 0.5 day | Usability |
| 5 | Load testing | 1 day | Reliability |
| 6 | Kubernetes manifests | 1 day | Deployment |
| 7 | Helm chart | 1 day | Deployment |

---

## Quick Wins

### Authentication Middleware
```go
func authMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        apiKey := r.Header.Get("Authorization")
        if apiKey == "" {
            apiKey = r.URL.Query().Get("api_key")
        }
        if !validateKey(apiKey) {
            http.Error(w, "Unauthorized", http.StatusUnauthorized)
            return
        }
        next.ServeHTTP(w, r)
    })
}
```

### CORS Middleware
```go
func corsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", os.Getenv("CORS_ORIGINS"))
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
        if r.Method == "OPTIONS" {
            w.WriteHeader(http.StatusOK)
            return
        }
        next.ServeHTTP(w, r)
    })
}
```

---

## Testing Strategy

### Unit Tests
- [ ] Engine adapter tests (mock engine)
- [ ] WebSocket handler tests
- [ ] Authentication middleware tests

### Integration Tests
- [ ] End-to-end inference flow
- [ ] Model hot-swapping
- [ ] Conversation history persistence

### Load Tests
- [ ] Concurrent WebSocket connections
- [ ] REST API throughput
- [ ] Memory leak detection

---

## Success Criteria

1. **Functionality**: WebUI can load models and generate text
2. **Performance**: <100ms latency for single-token generation
3. **Scalability**: Support 100+ concurrent connections
4. **Reliability**: 99.9% uptime target
5. **Security**: API key authentication required

---

## References

- Existing WebUI: `cmd/webui/`
- Engine API: `internal/engine/engine.go`
- CUDA backend: `internal/engine/engine_cuda.go`
- Metal backend: `internal/engine/engine.go` (darwin)
