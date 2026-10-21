#!/usr/bin/env bash
# Type-check the Apple Metal backend from a Linux host.
#
# The Metal backend is a `darwin && metal` build: it needs the macOS
# toolchain, so the only place it is normally compiled is the macos-latest
# CI runner. That runner did not run for months, and in the meantime the
# backend accumulated four compile errors - a missing import, an unused
# variable, a call with the wrong arity, and a lock on a field that did
# not exist. The `metal-tests` job was also gated behind a label, so
# nothing reported the result.
#
# The errors are all *Go type* errors, not errors that need the Metal
# framework. So they can be found on Linux: copy the tree, strip only the
# `darwin` token from the build tags, and build with cgo enabled. The
# cgo code itself is plain C (`metal_bridge.h`), so it compiles with the
# host compiler; only the *linking* of the final binary needs Metal, and
# this script never links a binary.
#
# What this checks:
#   - the Metal device and engine packages type-check
#   - the `darwin && metal` build tags still select the files they should
#
# What this does NOT check:
#   - that the Metal kernels produce correct numbers (see the Metal-vs-CPU
#     parity test in Step 60)
#   - that the cgo bridge links against the real Metal framework

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GO="${GO:-go}"
WORKDIR="${TMPDIR:-/tmp}/quarrel-metalcheck-$$"

PKGS="./internal/device/ ./internal/engine/ ./internal/gguf/"

cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

echo "=== Metal type check (darwin && metal, compiled on this host) ==="

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

# Copy the tree, but not the .git directory: it is large and the check
# does not need the history.
tar -C "$REPO_ROOT" --exclude=.git --exclude=bin --exclude=coverage \
    -cf - . | tar -C "$WORKDIR" -xf - || {
  echo "metalcheck: FAILED - could not copy the tree" >&2
  exit 1
}

# Strip only the `darwin` token from the build tags. The `metal` token
# stays, so the Metal files are still selected; the `cgo` tokens stay,
# so the cgo files are still compiled. What changes is that the host
# compiler is used instead of the macOS one, and that the `!darwin`
# fallbacks (which exist to catch a mis-tagged Metal build) are no longer
# selected.
python3 - "$WORKDIR" <<'PYEOF'
import os
import sys

root = sys.argv[1]
patched = 0
for dirpath, _dirnames, filenames in os.walk(root):
    for name in filenames:
        if not name.endswith(".go"):
            continue
        path = os.path.join(dirpath, name)
        with open(path, encoding="utf-8", errors="ignore") as fh:
            src = fh.read()
        if "darwin && metal" not in src:
            continue
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src.replace("darwin && metal", "metal"))
        patched += 1
if patched == 0:
    print("metalcheck: no `darwin && metal` build tags found", file=sys.stderr)
    sys.exit(1)
print(f"metalcheck: patched the build tags in {patched} files")
PYEOF
if [[ $? -ne 0 ]]; then
  echo "metalcheck: FAILED - could not patch the build tags" >&2
  exit 1
fi

cd "$WORKDIR" || exit 1

FAILED=0
for pkg in $PKGS; do
  out=$("$GO" build -gcflags=-e -tags metal "$pkg" 2>&1)
  if [[ $? -ne 0 ]]; then
    echo "--- $pkg"
    echo "$out" | sed 's/^/    /'
    FAILED=1
  fi
done

# The test files are type-checked too. `go build` does not compile them,
# so a Metal test file with a compile error was never noticed: the whole
# Metal test suite is behind a build tag that only macOS selects.
for pkg in $PKGS; do
  out=$("$GO" vet -tags metal "$pkg" 2>&1)
  if [[ $? -ne 0 ]]; then
    echo "--- $pkg (tests)"
    echo "$out" | sed 's/^/    /'
    FAILED=1
  fi
done

if [[ $FAILED -ne 0 ]]; then
  echo "metalcheck: FAILED - the Metal backend does not type-check"
  exit 1
fi

echo "metalcheck: OK - the Metal backend type-checks"
exit 0
