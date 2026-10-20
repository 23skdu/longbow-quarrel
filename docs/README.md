# Longbow-Quarrel Documentation

| Document | What it covers |
|----------|----------------|
| [usage.md](usage.md) | The command line, the options, the model resolution, the LoRA loading, the deployment |
| [api.md](api.md) | The two HTTP listeners, the endpoints, the request and response shapes |
| [architecture.md](architecture.md) | The layers of the engine, the element sizes, the directory layout |
| [features.md](features.md) | What the engine does, feature by feature |
| [comparison.md](comparison.md) | How it compares with llama.cpp, vLLM, Ollama and TensorRT-LLM |
| [metrics.md](metrics.md) | Every Prometheus metric the code declares |
| [performance.md](performance.md) | The measurements, with the method they were taken with |
| [dependencies.md](dependencies.md) | The pinned versions, and why each one is pinned |
| [testplan.md](testplan.md) | The A/B plan and the CI regression gates |
| [turboquant.md](turboquant.md) | The TurboQuant format and its kernels |
| [webui.md](webui.md) | The WebUI binary, its routes and its API |
| [README_CLOUD.md](README_CLOUD.md) | The cloud deployment: the limits, the probes, the Helm chart |
| [roadmap.md](roadmap.md) | The engineering roadmap: the open work, the gates, what has landed |
| [README.md](README.md) | This index |

`roadmap.md` is the one document that is not about the shipped product: it
records the work that is **not** finished, across the engine, the vector
search package and the WebUI.

The Helm chart lives in `helm/quarrel`, and the Grafana dashboard in
`grafana/quarrel-dashboard.json`.

## Keeping the documents honest

`scripts/docsync` checks that the documentation, the Helm chart and the
Grafana dashboard still describe the code, and it runs in the lint job and
as the twelfth gate of `scripts/run_all_tests.sh`. It checks that:

1. every flag the Helm chart passes is a flag the command has, and every
   value the chart sets matches the default the code uses;
2. every metric the Grafana dashboard queries is declared, in a query that
   matches the metric's type, and every declared metric is on the
   dashboard;
3. every flag the reference documentation mentions exists;
4. every endpoint the API reference lists is one the server registers, and
   every route the server registers is in the reference;
5. the pinned versions in `dependencies.md` are the ones the module files
   hold, and the number of metrics the documents claim is the real one;
6. every document is listed in this index.

```bash
go run ./scripts/docsync
go test ./scripts/docsync/
```
