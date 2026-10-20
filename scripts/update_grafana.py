#!/usr/bin/env python3
"""Bring the Grafana dashboard in line with the metrics the code declares.

The dashboard was written before most of the metrics existed: it queried
27 of the 133 declared ones, and the panels it did have queried a few
of them in the wrong shape -- a histogram read as if it were a gauge
shows the bucket counts rather than the value.

This script:

  * rewrites a panel that queries a histogram without the bucket suffix
    into the quantile form, and a panel that queries a gauge with a rate
    into the plain form;
  * adds a row for each metric family that was missing.

Run it from the repository root:

    python3 scripts/update_grafana.py

scripts/docsync.go checks that the result stays in line with the code.
"""

import json
import os
import re
import sys

DASHBOARD = "grafana/quarrel-dashboard.json"
METRICS_DIR = "internal/metrics"

DECL = re.compile(
    r"promauto\.New(?P<kind>Counter|CounterVec|Gauge|GaugeVec|Histogram|HistogramVec|Summary|SummaryVec)\("
    r"(?P<body>.*?)\n\s*\}",
    re.DOTALL,
)
NAME = re.compile(r'Name:\s*"(?P<name>[a-zA-Z0-9_:]+)"')

# The rows to add, in order. Each entry is a row title and the panels
# below it: a title, a unit and a list of (metric, legend) pairs.
ROWS = [
    (
        "Engine Health Audit",
        [
            (
                "Logit Range",
                "short",
                [
                    ("logit_max_value", "max"),
                    ("logit_min_value", "min"),
                    ("logit_mean_value", "mean"),
                    ("logit_rms", "rms"),
                ],
            ),
            (
                "Logit Anomalies",
                "short",
                [
                    ("logit_nan_count_total", "NaNs"),
                    ("logit_inf_count_total", "Infs"),
                    ("logit_flat_distribution_total", "flat"),
                    ("logit_extreme_values_total", "extreme"),
                ],
            ),
        ],
    ),
    (
        "Layer Offload and Memory",
        [
            (
                "Layer Offload",
                "s",
                [
                    ("quarrel_layer_offload_duration_seconds", "transfer"),
                    ("quarrel_layer_offload_transfers_total", "transfers"),
                ],
            ),
            (
                "Active Layers",
                "short",
                [
                    ("quarrel_cpu_layers_active", "cpu"),
                    ("quarrel_gpu_layers_active", "gpu"),
                ],
            ),
            (
                "Memory Pressure",
                "short",
                [
                    ("quarrel_memory_pressure_ratio", "ratio"),
                    ("quarrel_memory_pressure_events_total", "events"),
                    ("quarrel_memory_governor_actions_total", "actions"),
                ],
            ),
        ],
    ),
    (
        "Distributed Transfer",
        [
            (
                "Arrow Flight Transfer",
                "bytes",
                [
                    ("arrow_flight_bytes_transferred_total", "bytes"),
                    ("arrow_flight_embeddings_pushed_total", "embeddings"),
                    ("quarrel_distributed_transfer_bytes_total", "distributed bytes"),
                ],
            ),
            (
                "Distributed Transfer Duration",
                "s",
                [
                    ("quarrel_distributed_transfer_duration_seconds", "duration"),
                ],
            ),
        ],
    ),
    (
        "Mixture of Experts, Vision and the Tokenizer",
        [
            (
                "MoE Routing",
                "s",
                [
                    ("quarrel_moe_routing_latency_seconds", "routing"),
                    ("quarrel_moe_layer_latency_seconds", "layer"),
                    ("quarrel_moe_expert_selection_total", "selections"),
                ],
            ),
            (
                "VLM Encoding",
                "s",
                [
                    ("quarrel_vlm_image_encoding_duration_seconds", "encoding"),
                    ("quarrel_vlm_patches_processed_total", "patches"),
                ],
            ),
            (
                "Tokenizer",
                "s",
                [
                    ("tokenizer_encode_time_seconds", "encode"),
                    ("tokenizer_decode_time_seconds", "decode"),
                    ("tokenizer_encode_length", "length"),
                    ("tokenizer_unknown_tokens_total", "unknown tokens"),
                    ("tokenizer_vocab_size", "vocabulary"),
                ],
            ),
        ],
    ),
    (
        "SIMD Kernels",
        [
            (
                "SIMD Dispatch",
                "short",
                [
                    ("simd_dispatch_total", "dispatches"),
                    ("simd_fallback_count_total", "fallbacks"),
                    ("simd_level_detected", "level"),
                ],
            ),
            (
                "SIMD Kernel Duration",
                "s",
                [
                    ("simd_rmsnorm_duration_seconds", "rmsnorm"),
                    ("simd_rope_duration_seconds", "rope"),
                    ("simd_softmax_duration_seconds", "softmax"),
                    ("simd_swiglu_duration_seconds", "swiglu"),
                    ("simd_matmul_duration_seconds", "matmul"),
                    ("simd_attention_duration_seconds", "attention"),
                    ("simd_fused_mlp_duration_seconds", "fused mlp"),
                ],
            ),
        ],
    ),
    (
        "KV Cache Audit and Numerical Stability",
        [
            (
                "KV Cache Positions",
                "short",
                [
                    ("kv_cache_overlap_total", "overlaps"),
                    ("kv_cache_oob_total", "out of bounds"),
                    ("kv_cache_unique_positions", "unique"),
                ],
            ),
            (
                "Numerical Instability",
                "short",
                [
                    ("numerical_instability_total", "instabilities"),
                    ("nan_total_count", "NaNs"),
                    ("nan_layer_start", "first layer"),
                    ("nan_layer_end", "last layer"),
                ],
            ),
        ],
    ),
    (
        "Speculative Decoding Rate",
        [
            (
                "Speculative Tokens",
                "short",
                [
                    ("quarrel_speculative_draft_tokens_total", "drafted"),
                    ("quarrel_speculative_accepted_tokens_total", "accepted"),
                ],
            ),
            (
                "Speculative Acceptance",
                "short",
                [
                    ("quarrel_speculative_acceptance_rate", "ratio"),
                    ("quarrel_speculative_dynamic_draft_length", "draft length"),
                ],
            ),
        ],
    ),
    (
        "Batching Rates",
        [
            (
                "Batch State",
                "short",
                [
                    ("batch_queue_depth", "queue depth"),
                    ("batch_running_sequences", "running"),
                    ("quarrel_active_prefill_tokens", "prefill tokens"),
                    ("quarrel_continuous_batch_active_requests", "requests"),
                ],
            ),
            (
                "Batch Iterations",
                "s",
                [
                    ("quarrel_continuous_batch_iteration_duration_seconds", "iteration"),
                    ("quarrel_continuous_batch_preemptions_total", "preemptions"),
                ],
            ),
        ],
    ),
    (
        "Latency and Prefill",
        [
            (
                "Token Latency",
                "s",
                [
                    ("quarrel_time_to_first_token_seconds", "ttft"),
                    ("quarrel_inter_token_latency_seconds", "inter token"),
                ],
            ),
            (
                "Prefill",
                "s",
                [
                    ("quarrel_prefill_chunk_latency_seconds", "chunk"),
                    ("quarrel_flash_attention_prefill_duration_seconds", "flash attention"),
                    ("quarrel_flash_attention_sliding_window_tokens_total", "sliding window tokens"),
                ],
            ),
        ],
    ),
    (
        "RoPE and Softmax",
        [
            (
                "RoPE Accuracy",
                "short",
                [
                    ("rope_pass_total", "passes"),
                    ("rope_fail_total", "failures"),
                    ("rope_deviation", "deviation"),
                    ("rope_deviation_ratio", "deviation ratio"),
                ],
            ),
            (
                "Softmax Mask",
                "short",
                [
                    ("softmax_mask_value", "mask value"),
                ],
            ),
        ],
    ),
    (
        "Sampling Distribution",
        [
            (
                "Sampling Parameters",
                "short",
                [
                    ("sampling_top_p", "top p"),
                    ("sampling_top_token_probability", "top token probability"),
                    ("sampling_unique_samples", "unique samples"),
                    ("sampling_repetition_penalty", "repetition penalty"),
                ],
            ),
        ],
    ),
    (
        "SIMD Numerical Health",
        [
            (
                "SIMD Errors",
                "short",
                [
                    ("simd_nan_count_total", "NaNs"),
                    ("simd_inf_count_total", "Infs"),
                    ("simd_kernel_errors_total", "errors"),
                ],
            ),
            (
                "SIMD Kernel Duration",
                "s",
                [
                    ("simd_kernel_duration_seconds", "kernel"),
                    ("simd_kernel_duration_seconds", "kernel"),
                ],
            ),
        ],
    ),
    (
        "Memory Footprint",
        [
            (
                "Host and Device Memory",
                "bytes",
                [
                    ("host_memory_allocated_bytes", "host"),
                    ("quarrel_cuda_vram_saved_bytes", "vram saved"),
                ],
            ),
            (
                "KV Cache Compression",
                "short",
                [
                    ("quarrel_kv_cache_compression_ratio", "ratio"),
                    ("quarrel_kv_cache_quantized_pages_total", "quantized pages"),
                ],
            ),
        ],
    ),
    (
        "Activation Extremes",
        [
            (
                "Collapsed and Saturated Layers",
                "short",
                [
                    ("activation_collapsed_layers", "collapsed"),
                    ("activation_saturated_layers", "saturated"),
                    ("activation_residual_max", "residual max"),
                ],
            ),
        ],
    ),
    (
        "Grammar and Dequantization",
        [
            (
                "Grammar Filter",
                "s",
                [
                    ("quarrel_grammar_filter_duration_seconds", "filter"),
                    ("quarrel_grammar_constrained_tokens_total", "constrained tokens"),
                ],
            ),
            (
                "Dequantization Error",
                "short",
                [
                    ("dequant_max_abs_error", "absolute"),
                    ("dequant_max_rel_error", "relative"),
                    ("weight_padding_bytes", "weight padding"),
                    ("validation_errors_total", "validation errors"),
                ],
            ),
        ],
    ),
    (
        "Model Configuration",
        [
            (
                "Context and Heads",
                "short",
                [
                    ("context_length_tokens", "context"),
                    ("head_dim_threadgroup_size", "head dim threadgroup"),
                    ("buffer_gqa_ratio", "gqa ratio"),
                ],
            ),
            (
                "Gemma4 Configuration",
                "short",
                [
                    ("gemma4_context_length_tokens", "context"),
                    ("gemma4_sliding_window_size", "sliding window"),
                    ("gemma4_sliding_head_dim", "sliding head dim"),
                    ("gemma4_full_head_dim", "full head dim"),
                    ("gemma4_layer_pattern_ratio", "layer pattern"),
                    ("gemma4_partial_rope_factor", "partial rope factor"),
                    ("gemma4_rope_theta", "rope theta"),
                ],
            ),
        ],
    ),
    (
        "Attention and Expert Utilization",
        [
            (
                "Expert Utilization",
                "short",
                [
                    ("quarrel_moe_expert_utilization", "utilization"),
                ],
            ),
            (
                "Kernel Level",
                "s",
                [
                    ("quarrel_simd_vnni_dot_product_duration_seconds", "vnni dot"),
                    ("quarrel_cuda_dequant_gemm_duration_seconds", "cuda dequant gemm"),
                ],
            ),
        ],
    ),
]

PROMQL = {
    "rate", "increase", "sum", "avg", "max", "min", "count", "histogram_quantile",
    "le", "by", "without", "irate", "absent", "topk", "clamp", "stddev", "stdvar",
    "quantile", "delta", "idelta", "group",
}


def metric_types():
    """The metrics the code declares, with their Prometheus types."""
    types = {}
    for name in sorted(os.listdir(METRICS_DIR)):
        if not name.endswith(".go") or name.endswith("_test.go"):
            continue
        with open(os.path.join(METRICS_DIR, name)) as handle:
            source = handle.read()
        for match in DECL.finditer(source):
            body = match.group("body")
            name_match = NAME.search(body)
            if not name_match:
                continue
            kinds = {
                "Counter": "Counter", "CounterVec": "Counter",
                "Gauge": "Gauge", "GaugeVec": "Gauge",
                "Histogram": "Histogram", "HistogramVec": "Histogram",
                "Summary": "Summary", "SummaryVec": "Summary",
            }
            types[name_match.group("name")] = kinds[match.group("kind")]
    return types


def expression(metric, kind):
    """The PromQL that reads a metric the way its type needs."""
    if kind == "Histogram":
        return f"histogram_quantile(0.95, rate({metric}_bucket[5m]))"
    if kind == "Counter":
        return f"rate({metric}[5m])"
    return metric


def referenced(expr):
    """The metric names an expression refers to."""
    found = []
    for name in re.findall(r"\b([a-z][a-z0-9_]*)\b", expr):
        if name in PROMQL:
            continue
        if name.endswith(("_bucket", "_sum", "_count")) and name.rsplit("_", 1)[0]:
            found.append(name.rsplit("_", 1)[0])
            continue
        found.append(name)
    return found


def fix_expression(expr, types):
    """Rewrite an expression whose query form does not match the type."""
    names = [n for n in referenced(expr) if n in types]
    if not names:
        return expr, False

    kind = types[names[0]]
    uses_rate = "rate(" in expr or "increase(" in expr
    uses_bucket = "_bucket" in expr

    if kind == "Histogram" and not uses_bucket:
        return expression(names[0], kind), True
    if kind == "Gauge" and uses_rate:
        return names[0], True
    if kind == "Counter" and uses_bucket:
        return f"rate({names[0]}[5m])", True
    return expr, False


def timeseries(panel_id, title, unit, series, grid, types):
    targets = []
    for i, (metric, legend) in enumerate(series):
        targets.append(
            {
                "datasource": {"type": "prometheus", "uid": "${datasource}"},
                "expr": expression(metric, types.get(metric, "Gauge")),
                "legendFormat": legend,
                "refId": chr(ord("A") + i),
            }
        )
    return {
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "axisBorderShow": False,
                    "axisCenteredZero": False,
                    "axisColorMode": "text",
                    "axisLabel": "",
                    "axisPlacement": "auto",
                    "barAlignment": 0,
                    "drawStyle": "line",
                    "fillOpacity": 10,
                    "gradientMode": "none",
                    "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                    "insertNulls": False,
                    "lineInterpolation": "linear",
                    "lineWidth": 1,
                    "pointSize": 5,
                    "scaleDistribution": {"type": "linear"},
                    "showPoints": "auto",
                    "spanNulls": False,
                    "stacking": {"group": "A", "mode": "none"},
                    "thresholdsStyle": {"mode": "off"},
                },
                "mappings": [],
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "green", "value": None}],
                },
                "unit": unit,
            },
            "overrides": [],
        },
        "gridPos": grid,
        "id": panel_id,
        "options": {
            "legend": {
                "calcs": [],
                "displayMode": "list",
                "placement": "bottom",
                "showLegend": True,
            },
            "tooltip": {"mode": "single", "sort": "none"},
        },
        "pluginVersion": "13.2.3",
        "targets": targets,
        "title": title,
        "type": "timeseries",
    }


def row(panel_id, title, grid):
    return {
        "collapsed": False,
        "gridPos": grid,
        "id": panel_id,
        "panels": [],
        "title": title,
        "type": "row",
    }


def main():
    if not os.path.exists(DASHBOARD):
        print(f"{DASHBOARD} not found", file=sys.stderr)
        return 1

    types = metric_types()
    if not types:
        print("no metrics were found in the code", file=sys.stderr)
        return 1

    with open(DASHBOARD) as handle:
        dashboard = json.load(handle)

    panels = dashboard["panels"]

    # Repair the panels whose query form does not match the metric type.
    repaired = 0
    for panel in panels:
        for target in panel.get("targets", []):
            expr = target.get("expr", "")
            new_expr, changed = fix_expression(expr, types)
            if changed:
                target["expr"] = new_expr
                repaired += 1
    if repaired:
        print(f"repaired {repaired} panel queries")

    used_ids = {panel.get("id") for panel in panels}
    existing_rows = {p.get("title") for p in panels if p.get("type") == "row"}

    next_id = max(i for i in used_ids if isinstance(i, int)) + 1
    max_y = 0
    for panel in panels:
        grid = panel.get("gridPos") or {}
        max_y = max(max_y, grid.get("y", 0) + grid.get("h", 0))

    for title, panel_specs in ROWS:
        if title in existing_rows:
            print(f"the row {title!r} is already there")
            continue

        panels.append(row(next_id, title, {"h": 1, "w": 24, "x": 0, "y": max_y}))
        next_id += 1
        max_y += 1

        for i, (panel_title, unit, series) in enumerate(panel_specs):
            columns = 2 if len(panel_specs) > 1 else 1
            grid = {
                "h": 6,
                "w": 24 // columns,
                "x": (i % columns) * (24 // columns),
                "y": max_y,
            }
            panels.append(timeseries(next_id, panel_title, unit, series, grid, types))
            next_id += 1
            if (i + 1) % columns == 0 or columns == 1:
                max_y += 6

    dashboard["panels"] = panels
    dashboard["version"] = dashboard.get("version", 1) + 1

    with open(DASHBOARD, "w") as handle:
        json.dump(dashboard, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"the dashboard now has {len(panels)} panels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
