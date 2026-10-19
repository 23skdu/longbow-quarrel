#!/usr/bin/env bash
# Check every package for duplicate package-level declarations across the
# build configurations this repo supports.
#
# A duplicate top-level declaration is a compile error ("redeclared in this
# block"), but it only surfaces when two files that declare the same name end
# up in the same package. With build-tag-selected files that can hide in a
# configuration nobody builds locally: the darwin+metal combination had a
# duplicate ErrModelSwapped that broke the macOS Metal build while every
# Linux build stayed green.
#
# `go list` selects the same files the compiler would, and it does not need a
# C compiler, so a darwin or cuda configuration can be checked from Linux.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

GO="${GO:-go}"
FAILED=0
CHECKED=0

# The configurations the repo builds for. "tags" is passed through verbatim.
CONFIGS=(
  "linux|amd64|1|"
  "linux|amd64|0|"
  "linux|arm64|0|"
  "darwin|arm64|1|metal darwin"
  "darwin|amd64|1|metal darwin"
  "linux|amd64|1|cuda"
  "linux|amd64|1|avx512"
)

echo "=== Duplicate declaration check ==="

PKGS=$("$GO" list -f '{{.Dir}}' ./... 2>/dev/null)
if [[ -z "$PKGS" ]]; then
  echo "dupcheck: go list returned no packages" >&2
  exit 1
fi

for dir in $PKGS; do
  rel="${dir#$REPO_ROOT/}"
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r goos goarch cgo tags <<< "$cfg"
    CHECKED=$((CHECKED + 1))
    out=$("$GO" run ./scripts/dupcheck -dir "$rel" -tags "$tags" \
            -goos "$goos" -goarch "$goarch" -cgo "$cgo" 2>&1)
    if [[ $? -ne 0 ]]; then
      echo "--- $rel [$goos/$goarch cgo=$cgo tags='$tags']"
      echo "$out" | sed 's/^/    /'
      FAILED=1
    fi
  done
done

if [[ $FAILED -ne 0 ]]; then
  echo "dupcheck: FAILED - duplicate package-level declarations found"
  exit 1
fi

echo "dupcheck: OK - $CHECKED package/configuration combinations, no duplicates"
exit 0
