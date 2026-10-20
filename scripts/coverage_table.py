#!/usr/bin/env python3
"""Regenerate the coverage table in README.md from a fresh measurement.

The table used to be maintained by hand, and it drifted every time the
coverage moved: eight of its nineteen rows were wrong the last time
anybody checked. This script measures, renders and rewrites the table,
so the numbers in the README are the numbers the code produces.

The CI runs it with --check and fails if the working tree changed, so a
change that moves the coverage has to regenerate the table.

Usage:
    scripts/coverage_table.py            # rewrite the table in README.md
    scripts/coverage_table.py --check    # fail if the table is out of date
    scripts/coverage_table.py --print    # print the table and exit
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(REPO_ROOT, "README.md")
START = "<!-- coverage:start -->"
END = "<!-- coverage:end -->"


def run(*args: str) -> str:
    return subprocess.run(
        args, cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def measure() -> dict[str, float]:
    """The per-package coverage of the default build."""
    with tempfile.NamedTemporaryFile(suffix=".out", delete=False) as fh:
        profile = fh.name
    try:
        subprocess.run(
            ["go", "test", "-count=1", "-covermode=atomic",
             "-coverprofile=" + profile, "./internal/..."],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        statements: dict[str, int] = {}
        covered: dict[str, int] = {}
        with open(profile, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("mode:"):
                    continue
                # The line is "path:start,end numStmts count".
                parts = line.rsplit(" ", 2)
                if len(parts) != 3:
                    continue
                location, num, count = parts
                path = location.split(":", 1)[0]
                match = re.search(r"/(internal/.*)$", path)
                if not match:
                    continue
                package = os.path.dirname(match.group(1))
                try:
                    num, count = int(num), int(count)
                except ValueError:
                    continue
                statements[package] = statements.get(package, 0) + num
                if count > 0:
                    covered[package] = covered.get(package, 0) + num
    finally:
        os.unlink(profile)

    rows = []
    for line in run("go", "list", "./internal/...").split("\n"):
        match = re.search(r"/(internal/.*)$", line.strip())
        if not match:
            continue
        package = match.group(1)
        total = statements.get(package, 0)
        if total == 0:
            continue
        rows.append((package, 100.0 * covered.get(package, 0) / total))

    # Fully covered first, then the partial ones by ascending coverage.
    rows.sort(key=lambda row: (-row[1], row[0]))
    return dict(rows)


def caveat() -> str:
    """The note that says what the numbers do not cover."""
    def counts(directory: str) -> tuple[int, int]:
        files = []
        for name in os.listdir(os.path.join(REPO_ROOT, directory)):
            if not name.endswith(".go"):
                continue
            path = os.path.join(REPO_ROOT, directory, name)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                first = fh.readline()
            files.append(not first.startswith("//go:build"))
        return len(files), sum(files)

    device_files, device_untagged = counts("internal/device")
    engine_files, engine_untagged = counts("internal/engine")

    return (
        "> These numbers cover the default build only. `internal/device` has\n"
        "> %d `.go` files but only %d compile without a build tag, and\n"
        "> `internal/engine` has %d files of which %d compile — all CUDA,\n"
        "> Metal, TPU and AVX-512 code is excluded from both the numerator\n"
        "> and the denominator. The percentages therefore say nothing about\n"
        "> the GPU backends."
        % (device_files, device_untagged, engine_files, engine_untagged)
    )


def render(percentages: dict[str, float]) -> str:
    lines = ["| Package | Coverage |", "|---|---|"]
    for package, pct in percentages.items():
        lines.append("| `%s` | %.1f%% |" % (package, pct))
    return "\n".join(lines)


def rewrite(block: str) -> bool:
    """Writes the block into the README. Returns True when it changed."""
    with open(README, encoding="utf-8") as fh:
        readme = fh.read()
    if START not in readme or END not in readme:
        print("coverage_table: the README has no %s / %s markers" % (START, END),
              file=sys.stderr)
        sys.exit(1)
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    updated = pattern.sub(lambda _: block, readme, count=1)
    if updated == readme:
        return False
    with open(README, "w", encoding="utf-8") as fh:
        fh.write(updated)
    return True


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    percentages = measure()
    table = render(percentages)

    if mode == "--print":
        print(table)
        return 0

    block = "%s\n\n%s\n\n%s\n\n%s" % (START, caveat(), table, END)
    changed = rewrite(block)
    if changed:
        print("coverage_table: the README table was regenerated")
    else:
        print("coverage_table: the README table is up to date")

    if mode == "--check":
        if changed:
            print("coverage_table: FAILED - README.md is out of date; "
                  "run scripts/coverage_table.py", file=sys.stderr)
            return 1
        print("coverage_table: OK - the README table matches the measurement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
