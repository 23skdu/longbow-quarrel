# Pinned Dependency Versions

Single source of truth for every version pin in the repository. Last reviewed:
**October 2026**.

Refresh procedure:

```bash
go list -m -u all                       # available updates for the root module
go get <module>@<version> && go mod tidy
(cd cmd/webui && go mod tidy)           # second module, must be kept in sync
go run ./scripts/vacuous_tests          # unrelated but part of the gate
./scripts/dupcheck.sh                   # duplicate top-level declarations
```

---

## Go module — root (`github.com/23skdu/longbow-quarrel`)

Toolchain: `go 1.27.2`

| Module | Version | Notes |
|---|---|---|
| `github.com/apache/arrow-go/v18` | v18.8.0 | current |
| `github.com/mxschmitt/playwright-go` | v0.6201.1 | browser e2e test, see below |
| `github.com/prometheus/client_golang` | v1.25.0 | current |
| `github.com/prometheus/client_model` | v0.6.3 | current |
| `github.com/rs/zerolog` | v1.35.1 | current |
| `go.opentelemetry.io/otel` (+ exporters, sdk, trace, metric) | v1.47.0 | current |
| `golang.org/x/image` | v0.47.0 | current |
| `google.golang.org/grpc` | v1.86.0-dev | security pin, see below |
| `github.com/emlgo/eml` (replaced by `github.com/23skdu/emlgo`) | v0.0.0-20261015062852-39bcca0c1034 | math backend, see below; **optional, `-tags emlgo`** |

> The versions above are the ones in `go.mod` and `go.sum`. They were
> regenerated from the module files rather than maintained by hand.

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

### Why `playwright-go` uses the upstream import path

The repository previously used the community fork
`github.com/playwright-community/playwright-go` (v0.6000.0), which was the
newest release importable under that path: upstream reverted the module path
at v0.6100.0, so requesting v0.6100.0+ under the community path fails with

```
module declares its path as: github.com/mxschmitt/playwright-go
        but was required as: github.com/playwright-community/playwright-go
```

**The fork could no longer download its driver.** The community fork still
points at the retired `playwright.azureedge.net` CDN, so every driver
download 404s:

```
error: got non 200 status code: 404 (Not Found) from
  https://playwright.azureedge.net/builds/driver/playwright-1.60.0-linux.zip
```

Because the e2e test skips itself when the driver is missing, the browser
test had **never run** -- it was silently skipped in every CI run.

The import path is now the maintained upstream,
`github.com/mxschmitt/playwright-go` v0.6201.1, which downloads its driver
from the current CDN. The CI `build` job installs the driver so the test
actually executes. Playwright is used only by
`internal/api/playwright_e2e_test.go`.

### Why `golang-set/v2` is held at v2.8.0

v2.9.0 adds a `go.mongodb.org/mongo-driver` requirement to its `go.mod`.
Adopting it would pull a MongoDB driver into the build graph of an inference
engine for no functional gain. Held at v2.8.0; revisit only if upstream
restructures its test dependencies.

### Why `emlgo` requires a fork (module-path mismatch)

`emlgo` is consumed as an **optional** math backend (see
[roadmap.md](roadmap.md#planned-features)) and is **not** a drop-in `math`
replacement. Two things must be understood before adding it:

1. **Module path mismatch.** The repository is hosted at
   `github.com/23skdu/emlgo`, but its `go.mod` declares
   `module github.com/emlgo/eml` and every internal import uses the
   `github.com/emlgo/eml/...` path. Requesting it under the host path fails:

   ```
   $ go get github.com/23skdu/emlgo@0.5
   module declares its path as: github.com/emlgo/eml
           but was required as: github.com/23skdu/emlgo
   ```

   A fork with a corrected `module` line (and rewritten internal imports) is
   therefore required. The pin above refers to that fork.
2. **No semver tag.** The upstream release is tagged `0.7`, which Go does
   not treat as semver. The pinned version is the pseudo-version
   `v0.0.0-20261015062852-39bcca0c1034` (commit
   `39bcca0c1034c51cd7e054daa60b9059f0c3846d`), which corresponds to the
   upstream tag `0.7` / `latest`.

`emlgo` is only compiled in with `-tags emlgo`; the default build is
unaffected. Its scalar `pkg/*` functions delegate to stdlib `math` (bit-exact);
`pkg/fastmath` uses minimax polynomial approximations (not bit-exact) and is
gated separately behind `-tags emlfast`.

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
| `golang` | `1.27.2-alpine3.24` | current 1.27 patch |
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