#!/bin/bash
# Resolve the path to a runnable longbow-quarrel binary.
#
# The benchmark and profiling scripts used to look for
#   bin/quarrel-cuda  and  bin/quarrel-simple
# but the Makefile has never produced either name -- it emits
#   bin/quarrel-linux-amd64-cuda
#   bin/quarrel-linux-amd64
#   bin/quarrel-linux-amd64-avx512
#   bin/quarrel-linux-arm64
#   bin/quarrel-darwin-arm64
# so on a clean checkout the scripts always failed with "binary not found".
#
# Resolution order:
#   1. $QUARREL_BIN, if set and executable
#   2. a prebuilt binary in bin/ matching this platform
#   3. `go run ./cmd/quarrel` as a fallback, so the scripts work straight
#      from a source checkout
#
# Source this file; call resolve_quarrel_bin.

# resolve_quarrel_bin sets QUARREL_BIN to a usable command.
resolve_quarrel_bin() {
    if [[ -n "${QUARREL_BIN:-}" && -x "${QUARREL_BIN}" ]]; then
        return 0
    fi

    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    local os arch
    case "$(uname -s)" in
        Darwin) os="darwin" ;;
        Linux)  os="linux" ;;
        *)      os="$(uname -s | tr '[:upper:]' '[:lower:]')" ;;
    esac
    case "$(uname -m)" in
        x86_64|amd64) arch="amd64" ;;
        arm64|aarch64) arch="arm64" ;;
        *)             arch="$(uname -m)" ;;
    esac

    local candidates=(
        # Prefer the plain, platform-native build: a CUDA binary cannot run
        # on a host without an NVIDIA GPU, and a Metal binary cannot run
        # off macOS.
        "$repo_root/bin/quarrel-${os}-${arch}"
        "$repo_root/bin/quarrel-linux-amd64"
        "$repo_root/bin/quarrel-${os}-${arch}-cuda"
        "$repo_root/bin/quarrel-linux-amd64-cuda"
        "$repo_root/bin/quarrel"
    )

    for c in "${candidates[@]}"; do
        if [[ -x "$c" ]]; then
            QUARREL_BIN="$c"
            return 0
        fi
    done

    # No prebuilt binary. Build one so the scripts work straight from a
    # source checkout. The result is a single executable path, so callers can
    # keep using "$QUARREL_BIN" as a plain command.
    if command -v go >/dev/null 2>&1; then
        local built="$repo_root/bin/quarrel-bench"
        if mkdir -p "$repo_root/bin" && \
           (cd "$repo_root" && go build -o "$built" ./cmd/quarrel) ; then
            QUARREL_BIN="$built"
            return 0
        fi
        echo "error: failed to build the longbow-quarrel binary." >&2
        echo "  Build it manually with: make cpu-linux-amd64" >&2
        return 1
    fi

    echo "error: no longbow-quarrel binary found and go is not available." >&2
    echo "  Build one with: make cpu-linux-amd64   (or: make nvidia)" >&2
    echo "  Or set QUARREL_BIN to an existing binary." >&2
    return 1
}

# run_quarrel runs the resolved binary with the given arguments.
run_quarrel() {
    resolve_quarrel_bin || return 1
    # shellcheck disable=SC2086
    $QUARREL_BIN "$@"
}
