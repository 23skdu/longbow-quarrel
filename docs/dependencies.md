# Pinned Dependency Versions

Single source of truth for every version pin in the repository. Last reviewed:
**September 2026**.

Refresh procedure:

```bash
go list -m -u all                       # available updates for the root module
go get <module>@<version> && go mod tidy
(cd cmd/webui && go mod tidy)           # second module, must be kept in sync
go run ./scripts/vacuous_tests          # unrelated but part of the gate
```

---

## Go module — root (`github.com/23skdu/longbow-quarrel`)

Toolchain: `go 1.27.0`

| Module | Version | Update available | Notes |
|---|---|---|---|
| `github.com/apache/arrow-go/v18` | v18.8.0 | no | current |
| `github.com/playwright-community/playwright-go` | v0.6000.0 | yes, but see note | capped, see below |
| `github.com/prometheus/client_golang` | v1.24.1 | no | current |
| `github.com/prometheus/client_model` | v0.6.3 | no | current |
| `github.com/rs/zerolog` | v1.35.1 | no | current |
| `go.opentelemetry.io/otel` (+ exporters, sdk, trace, metric) | v1.46.0 | v1.47.0-rc.1 | held: RC only |
| `golang.org/x/image` | v0.46.0 | no | current |
| `google.golang.org/grpc` | v1.85.0-dev.0.20260825072537-93e31b48545e | yes, but see note | security pin, see below |

### Why `grpc` is pinned to a pseudo-version

`google.golang.org/grpc` is pinned to the **exact commit that fixes
[GO-2026-6443](https://pkg.go.dev/vuln/GO-2026-6443)** / CVE-2026-84445 —
a server panic triggered by requests missing both `:authority` and `Host`
headers under xDS routing.

The advisory's affected ranges are:

| Introduced | Fixed in |
|---|---|
| `0` | `1.82.2` |
| `1.83.0` | `1.83.2` |
| `1.84.0-dev` | `1.85.0-dev.0.20260825072537-93e31b48545e` |

No stable tag (`v1.85.0`, `v1.86.0`) exists yet, so moving to a release tag
would reintroduce the vulnerability. `v1.86.0-dev` is newer and also contains
the fix, but staying on the documented fix commit keeps the pin explainable.
**Re-check this row once gRPC cuts a stable release**, and confirm with:

```bash
govulncheck ./...
```

### Why `playwright-go` is capped at v0.6000.0

Upstream reverted the module path at v0.6100.0. The declared `module` line is:

| Version | Declared module path |
|---|---|
| v0.5700.0 – v0.6000.0 | `github.com/playwright-community/playwright-go` |
| v0.6100.0 and later | `github.com/mxschmitt/playwright-go` |

Requesting v0.6100.0+ under the community path fails:

```
module declares its path as: github.com/mxschmitt/playwright-go
        but was required as: github.com/playwright-community/playwright-go
```

v0.6000.0 is therefore the newest release importable under the path this
repository uses. Adopting v0.6201.1 requires switching the import path to
`github.com/mxschmitt/playwright-go`, which is a deliberate decision rather
than a version bump. Playwright is used only by `internal/api/playwright_e2e_test.go`.

### Why `golang-set/v2` is held at v2.8.0

v2.9.0 adds a `go.mongodb.org/mongo-driver` requirement to its `go.mod`.
Adopting it would pull a MongoDB driver into the build graph of an inference
engine for no functional gain. Held at v2.8.0; revisit only if upstream
restructures its test dependencies.

---

## Go module — `cmd/webui` (separate module)

`cmd/webui` is its own module and uses `replace github.com/23skdu/longbow-quarrel => ../..`,
so it must be tidied separately. It was a full version behind the root module
(arrow-go 18.7.0, lz4 4.1.29, x/net 0.58.0, x/sys 0.47.0, x/text 0.41.0,
genproto Sept-04 pseudo-versions, grpc 1.83.2) and is now aligned with the root.

Because `go build ./...` from the repository root does **not** descend into
`cmd/webui`, CI runs a dedicated job for it. Locally:

```bash
cd cmd/webui && go build ./... && go vet ./... && go test ./...
```

---

## CI toolchain (`.github/workflows/ci.yml`)

| Tool | Version | Notes |
|---|---|---|
| Go | `1.27` | matches `go.mod` |
| golangci-lint | v2.14.0 | must stay v2: repo config is `version: "2"` |
| gosec | v2.29.0 | current |
| goimports | v0.50.0 | tracks the `golang.org/x/tools` series |
| govulncheck | v1.8.0 | current |

### GitHub Actions

Pinned to latest majors. All of these run on the Node 24 runtime and require
a runner **≥ 2.329.0**.

| Action | Version |
|---|---|
| `actions/checkout` | v7 |
| `actions/setup-go` | v7 |
| `actions/cache` | v6 |
| `actions/upload-artifact` | v7 |
| `codecov/codecov-action` | v7 |
| `docker/setup-buildx-action` | v4 |
| `docker/build-push-action` | v7 |
| `docker/login-action` | v4 |
| `softprops/action-gh-release` | v3 |

> **Self-hosted runner caveat:** the GPU job runs on
> `runs-on: [self-hosted, gpu]`. That runner must be updated to ≥ 2.329.0 or
> the job fails at the checkout step.

`actions/upload-artifact` v5+ requires artifact names to be unique within a
workflow run. The fuzz job satisfies this via `matrix.target`/`matrix.duration`.

---

## Container images

| Image | Pinned to | Rationale |
|---|---|---|
| `golang` | `1.27.1-alpine3.24` | current 1.27 patch |
| `alpine` | `3.24` | 3.19 is EOL |
| `nvidia/cuda` | `12.9.2-devel-ubuntu24.04`, `12.9.2-runtime-ubuntu24.04` | latest CUDA 12.x |
| `ubuntu` | `24.04` (Dockerfile.tpu) | jammy has no modern Python |
| `prom/prometheus` | `v3.15.0` | v2 series is EOL |
| `grafana/grafana` | `13.2.3` | matches dashboard `pluginVersion` |

### CUDA major version

Held at **12.9.x**. CUDA 13.x is available but is a major toolkit bump; the
repo's `.cu` sources only need `cuda_runtime.h` and `cuda_fp16.h`, so 12.9 is
sufficient and keeps the `libcudnn9-cuda-12` package names valid. CUDA 12.9
deprecates Maxwell/Pascal/Volta offline compilation (removal in the next major
toolkit), so a future CUDA 13 move also needs a GPU-support decision.

The local development toolkit for this repository is CUDA 12.4 (`nvcc`); the
container images run a newer 12.x patch.

### JAX / TPU

`Dockerfile.tpu` pins `jax[tpu]==0.11.2`. This requires Python ≥ 3.12, which is
why the image is based on Ubuntu 24.04 (Python 3.12) rather than 22.04
(Python 3.10). The previously pinned `jax==0.4.26` and the extra index
`https://storage.googleapis.com/jax-releases/jax_lib.html` were both removed:
the former is long superseded, the latter now returns HTTP 404.

---

## Helm chart

`helm/quarrel/Chart.yaml` tracks `0.3.0` to match the latest git tag. Bump the
chart `version` and `appVersion` together when cutting a release tag.