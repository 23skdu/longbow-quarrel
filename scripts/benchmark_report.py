#!/usr/bin/env python3
"""
benchmark_report.py — Generate comparison report from benchmark CSV/JSON results.

Reads benchmark CSV files produced by benchmark_engines.sh / benchmark_models.sh
and produces a formatted comparison table.

Usage:
    python3 scripts/benchmark_report.py benchmark_results/summary.csv
    python3 scripts/benchmark_report.py benchmark_results/*.csv --format json
    python3 scripts/benchmark_report.py benchmark_results/summary.csv --best
"""

import csv
import json
import sys
import os
from collections import defaultdict
from typing import Optional

def load_csv(filepath: str) -> list[dict]:
    results = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row['tps_mean'] = float(row.get('tps_mean', 0))
                row['tps_stddev'] = float(row.get('tps_stddev', 0))
                row['tps_min'] = float(row.get('tps_min', 0))
                row['tps_max'] = float(row.get('tps_max', 0))
                row['tokens'] = int(row.get('tokens', 0))
                row['runs'] = int(row.get('runs', 0))
            except (ValueError, KeyError):
                continue
            results.append(row)
    return results


def group_by_model(results: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for r in results:
        grouped[r['model']].append(r)
    return dict(grouped)


def format_table(headers: list[str], rows: list[list[str]], alignments: Optional[list[str]] = None) -> str:
    if not rows:
        return "(no data)"

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(cells, aligns=None):
        parts = []
        for i, cell in enumerate(cells):
            w = widths[i]
            a = (aligns or ['left'] * len(cells))[i] if aligns else 'left'
            if a == 'right':
                parts.append(str(cell).rjust(w))
            elif a == 'center':
                parts.append(str(cell).center(w))
            else:
                parts.append(str(cell).ljust(w))
        return ' │ '.join(parts)

    sep_parts = ['─' * w for w in widths]
    sep = '─┼─'.join(sep_parts)

    lines = [fmt_row(headers)]
    lines.append(sep)
    for row in rows:
        lines.append(fmt_row(row, alignments))

    return '\n'.join(lines)


def print_comparison(results: list[dict], show_best: bool = False):
    grouped = group_by_model(results)

    for model, model_results in sorted(grouped.items()):
        engines = sorted(set(r['engine'] for r in model_results))
        tokens = model_results[0]['tokens'] if model_results else 0

        print(f"\n{'='*60}")
        print(f"Model: {model} ({tokens} tokens)")
        print(f"{'='*60}")

        headers = ['Engine', 'Mean t/s', 'StdDev', 'Min', 'Max', 'Runs']
        rows = []
        best_tps = 0
        best_engine = ""

        for r in sorted(model_results, key=lambda x: x['tps_mean'], reverse=True):
            tps = r['tps_mean']
            if tps > best_tps:
                best_tps = tps
                best_engine = r['engine']
            rows.append([
                r['engine'],
                f"{tps:.2f}",
                f"{r['tps_stddev']:.2f}",
                f"{r['tps_min']:.2f}",
                f"{r['tps_max']:.2f}",
                str(r['runs']),
            ])

        print(format_table(headers, rows, ['left', 'right', 'right', 'right', 'right', 'right']))

        if show_best and len(engines) > 1:
            for r in model_results:
                if r['engine'] != best_engine and best_tps > 0:
                    ratio = r['tps_mean'] / best_tps * 100
                    print(f"  {r['engine']} is at {ratio:.1f}% of {best_engine}")

        print()


def print_json(results: list[dict]):
    grouped = group_by_model(results)
    output = {}
    for model, model_results in grouped.items():
        output[model] = {}
        for r in model_results:
            output[model][r['engine']] = {
                'tps_mean': r['tps_mean'],
                'tps_stddev': r['tps_stddev'],
                'tps_min': r['tps_min'],
                'tps_max': r['tps_max'],
                'runs': r['runs'],
                'tokens': r['tokens'],
            }
    print(json.dumps(output, indent=2))


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 benchmark_report.py <csv_files...> [--format json|table] [--best]")
        sys.exit(1)

    show_best = '--best' in sys.argv
    fmt = 'json' if '--json' in sys.argv or '--format json' in sys.argv else 'table'

    csv_files = [f for f in sys.argv[1:] if not f.startswith('--')]

    all_results = []
    for f in csv_files:
        if not os.path.exists(f):
            print(f"Warning: {f} not found", file=sys.stderr)
            continue
        all_results.extend(load_csv(f))

    if not all_results:
        print("No benchmark results found.", file=sys.stderr)
        sys.exit(1)

    if fmt == 'json':
        print_json(all_results)
    else:
        print_comparison(all_results, show_best)


if __name__ == '__main__':
    main()
