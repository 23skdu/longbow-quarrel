#!/usr/bin/env python3
"""List the metrics the code declares, with their Prometheus types.

    python3 scripts/list_metrics.py

The output is one "name type" pair per line. It is used to check the
Grafana dashboard against the code: a panel that queries a metric that
does not exist shows nothing, and a panel that queries a histogram as if
it were a gauge shows the bucket counts.
"""

import os
import re
import sys

METRICS_DIR = "internal/metrics"

# promauto.New<Kind> and the Name that follows it.
DECL = re.compile(
    r"promauto\.New(?P<kind>Counter|CounterVec|Gauge|GaugeVec|Histogram|HistogramVec|Summary|SummaryVec)\("
    r"(?P<body>.*?)\n\s*\}",
    re.DOTALL,
)
NAME = re.compile(r'Name:\s*"(?P<name>[a-zA-Z0-9_:]+)"')


def main():
    if not os.path.isdir(METRICS_DIR):
        print(f"{METRICS_DIR} not found", file=sys.stderr)
        return 1

    metrics = {}
    for name in sorted(os.listdir(METRICS_DIR)):
        if not name.endswith(".go") or name.endswith("_test.go"):
            continue
        with open(os.path.join(METRICS_DIR, name)) as handle:
            source = handle.read()

        for match in DECL.finditer(source):
            kind = match.group("kind")
            body = match.group("body")
            name_match = NAME.search(body)
            if not name_match:
                continue
            metric = name_match.group("name")
            kind = kind.replace("Vec", "")
            metrics[metric] = kind

    for metric, kind in sorted(metrics.items()):
        print(f"{metric} {kind}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
