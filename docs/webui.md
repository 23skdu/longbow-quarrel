# WebUI

> **Scope.** This document describes the WebUI binary: how it is built,
> how it is configured, what it serves. It used to be a forward-looking
> plan (473 lines, every checkbox unchecked) that proposed a stack the code
> does not use: `github.com/a-h/templ` for templating,
> `github.com/gin-gonic/gin` for routing and `github.com/markbates/grift`
> for CLI tooling. None of them are dependencies of `cmd/webui`.
>
> The ideas that are still open are in
> [roadmap.md](roadmap.md#planned-features).

## 1. Overview

`cmd/webui` is a small HTTP server that exposes the inference engine over a
browser UI and a JSON API. It is a **separate Go module** (`cmd/webui/go.mod`)
so it can be built and deployed independently of the engine.

The stack is deliberately minimal:

| Concern | Implementation |
|---|---|
| HTTP routing | `net/http` (`http.ServeMux`) |
| Templating | `html/template` (stdlib), composed from parts |
| WebSocket | `github.com/gorilla/websocket` |
| Metrics | `github.com/prometheus/client_golang` |
| Front-end | Plain HTML, CSS and JavaScript served from `cmd/webui/static/` |
| Browser tests | `@playwright/test` |

### No framework, and no build step

The front-end is plain JavaScript. A component framework needs a
bundler, and the WebUI is shipped as a static directory of files that a
Go binary serves: the cost of a framework is a `node_modules` directory
in the image and a build step in every release, and the benefit is a
component model the three hundred lines of `static/js/ui.js` do not need
yet.

The page is composed from parts for the same reason.
`cmd/webui/templates/index.go` holds it as five — the head, the header,
the sidebar, the chat and the scripts — parsed into one template set,
instead of one string in which a change to the sidebar meant finding the
right place by eye.

The model gallery and the LoRA adapter browser are in the sidebar, drawn
by `UIController.renderGallery` and `UIController.renderAdapters` from
`/api/models` and `/api/adapters`.

## 2. Build and run

```bash
# From inside cmd/webui (it is a separate module).
cd cmd/webui
go build -tags webui ./...
go run -tags webui .

# Or with the Docker image.
docker compose -f docker-compose.webui.yml up
```

Every file in the module is behind `//go:build webui`, so the tag is
required.

## 3. Configuration

Flags take precedence over the environment, and the environment over the
built-in defaults.

| Flag | Environment variable | Default |
|---|---|---|
| `-port` | `WEBUI_PORT` | `8080` |
| `-metrics-port` | `WEBUI_METRICS_PORT` | `9090` |
| `-host` | `WEBUI_HOST` | `0.0.0.0` |
| `-api-key` | `WEBUI_API_KEY` | *(empty — unauthenticated)* |
| `-allowed-origins` | `WEBUI_ALLOWED_ORIGINS` | *(empty — permissive)* |
| — | `WEBUI_MODEL_PATH` | *(empty)* |

`cmd/webui/config/config.go` implements the overlay;
`config_test.go` covers both directions, including that an unset variable
never clobbers an explicitly set value.

> **Security note.** With no API key configured the API is unauthenticated.
> The server logs a warning at startup when that is the case.

## 4. Routes

Registered in `cmd/webui/main.go`.

### Unauthenticated

| Route | Purpose |
|---|---|
| `GET /` | The single-page UI |
| `GET /static/…` | CSS and JavaScript |
| `GET /health`, `GET /healthz`, `GET /readyz` | Liveness and readiness probes |
| `GET /version` | Build version |
| `GET /metrics` | Prometheus metrics |
| `GET /debug/pprof/…` | Profiling endpoints |
| `GET /ws` | WebSocket streaming |

### Under `/api/`, API-key protected

| Route | Purpose |
|---|---|
| `GET /api/models` | List the loaded models |
| `GET /api/adapters` | List the LoRA adapters of the loaded models |
| `POST /api/generate` | Generate a completion |
| `POST /api/stream` | Stream a completion (SSE) |
| `POST /api/hotswap` | Swap the loaded model |
| `POST /api/v1/chat/completions` | OpenAI-compatible chat |
| `POST /api/v1/completions` | OpenAI-compatible completion |
| `GET /api/v1/models` | OpenAI-compatible model list |

The OpenAI-compatible routes are mounted under `/api/`, so the full path is
`/api/v1/chat/completions`. The engine's own REST API
(`internal/api`, served by `cmd/quarrel` on `-api-addr`) serves the same
paths **without** the `/api` prefix.

## 5. Backends

`cmd/webui/engine` selects an adapter by build tag:

| Adapter | Build tag | Backend |
|---|---|---|
| `adapter_cuda.go` | `linux && amd64 && cuda && cgo` | CUDA |
| `adapter_metal.go` | `darwin && metal` | Apple Metal |
| `adapter_cpu.go` | everything else | CPU |

All three run real inference. The CPU adapter loads a model with
`engine.NewEngine` and streams tokens the same way the GPU adapters do.

### The inference protocol

The caller does:

```go
outer, err := adapter.Infer(ctx, req)
inner := <-outer
for resp := range inner { … }
```

so the adapter must **deliver** the inner channel on the outer one. This is
easy to get wrong: an adapter that returns an outer channel nothing writes
to makes the request hang forever, and one that *takes* the inner channel
back out of the outer one starves the caller.

`cmd/webui/engine/adapter_protocol_test.go` is the regression guard: it
asserts that the inner channel is delivered, that the response is marked
complete, and that a failure carries an error rather than looking like an
empty completion.

## 6. Observability

`cmd/webui/handlers/metrics.go` exports five metrics:

| Metric | Type |
|---|---|
| `quarrel_webui_requests_total` | Counter |
| `quarrel_webui_errors_total` | Counter |
| `quarrel_webui_tokens_total` | Counter |
| `quarrel_webui_inference_duration_seconds` | Histogram |
| `quarrel_webui_connections_active` | Gauge |

They are listed in `docs/metrics.md`, which is checked against the source
by `internal/metrics/docs_test.go`.

## 7. Testing

```bash
cd cmd/webui
go vet  -tags webui ./...
go test -tags webui ./...
```

### The browser tests

The browser tests are in `cmd/webui/tests` and run with Playwright:

```bash
cd cmd/webui/tests
npm ci
npx playwright install --with-deps chromium
npx playwright test
```

They start the server themselves (the `webServer` block of
`playwright.config.ts`) and run against it. The CI job
`webui-browser` does the same.

The tests used to be in the repository and unable to run: there was no
`package.json`, the configuration pointed at a directory that did not
exist and at a port the server did not listen on, and nothing started
the server. They had never run, and they found four defects when they
did: every API endpoint answered 404 (the router mounted the API under
`/api/` without stripping the prefix), the settings were not saved, the
conversation was never read back from the local storage, and the
WebUI's own metrics were never counted.

The Go suite covers the same binary from the other side.

| Suite | What it covers |
|---|---|
| `config/config_test.go` | The environment overlay |
| `engine/adapter_protocol_test.go` | The inference protocol |
| `handlers/handlers_test.go` | The HTTP handlers |
| `handlers/fuzz_test.go` | Fuzzing of the request parsing |
| `tests/e2e/e2e_test.go` | End-to-end HTTP behaviour |
| `routes_test.go` | The routes, through the real router |
| `tests/*.spec.ts` | The browser tests (Playwright) |

The e2e suite used to be excluded from CI because it did not compile — it
referenced `handlers.Config` and `handlers.ModelInfo`, which had moved to
`cmd/webui/config` and `cmd/webui/engine`. It now compiles, runs, and is
part of the CI `webui` job. It found three latent defects when it was
enabled: a missing request body size limit, a rate limiter that ignored its
configured value, and the broken inference protocol.

## 8. See also

- [architecture.md](architecture.md) — the whole repository
- [api.md](api.md) — the OpenAI-compatible API
- [metrics.md](metrics.md) — the metric reference
- [roadmap.md](roadmap.md) — the open work
