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
| [webui_plan.md](webui_plan.md) | The WebUI binary, its routes and its API |
| [README_CLOUD.md](README_CLOUD.md) | The cloud deployment: the limits, the probes, the Helm chart |
| [nextsteps.md](nextsteps.md) | The engineering roadmap: what has landed and what is planned |
| [roadmap.md](roadmap.md) | The vector search subsystem, which is a separate component |
| [README.md](README.md) | This index |

Two of the documents describe a development log rather than the shipped
product: `nextsteps.md` is the engineering roadmap, and `roadmap.md`
covers the vector search package, which the inference path does not use.

The Helm chart lives in `helm/quarrel`, and the Grafana dashboard in
`grafana/quarrel-dashboard.json`.

## Keeping the documents honest

`scripts/docsync` checks that the documentation, the Helm chart and the
Grafana dashboard still describe the code, and it runs in the lint job and
as the thirteenth gate of `scripts/run_all_tests.sh`. It checks that:

1. every flag the Helm chart passes is a flag the command has, and every
   value the chart sets matches the default the code uses;
2. every metric the Grafana dashboard queries is declared, in a query that
   matches the metric's type, and every declared metric is on the
   dashboard;
3. every flag the reference documentation mentions exists;
4. every endpoint the API reference lists is one the server registers, and
   every route the server registers is in the reference;
5. the pinned versions in `dependencies.md` are the ones the module files
   hold, and the number of metrics the documents claim is the real one.

```bash
go run ./scripts/docsync
go test ./scripts/docsync/
```
