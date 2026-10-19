#!/usr/bin/env bash
# Verify the coverage gates from a single Go coverage profile.
#
# The CI coverage job used to run the test suite a second and third time
# (once per package) to collect the percentages, even though it had already
# produced coverage.out. That was slow, and the extraction was fragile:
#
#   go test -cover ... | grep -o 'coverage: [0-9.]*%' | awk '{print $2}'
#
# takes only the FIRST match when a pattern expands to several packages, and
# the `|| true` on the pipeline meant a *failed* test run also produced an
# empty string -- which then surfaced as a confusing "produced no coverage
# result" instead of the real failure.
#
# This script reads the profile directly. The coverage.out format is:
#
#   mode: atomic
#   <file>:<start>.<end> <numStatements> <count>
#
# where a count of 0 means the block was not covered. Summing the statement
# and covered counts per package gives the package coverage, with no second
# test run and no ambiguity.
#
# Usage: scripts/coverage_gate.sh <coverage-profile> [floors-file]

set -uo pipefail

PROFILE="${1:-coverage.out}"

if [[ ! -f "${PROFILE}" ]]; then
    echo "coverage_gate: profile ${PROFILE} not found" >&2
    exit 1
fi

# The packages that must meet the 80% threshold.
CORE_PKGS=(
    "./internal/sampler/..."
    "./internal/simd/..."
    "./internal/tokenizer/..."
    "./internal/metrics/..."
    "./internal/vlm/..."
    "./internal/gguf/..."
    "./internal/api/..."
    "./internal/vector/..."
)

# Packages below 80% are held to a ratchet floor: coverage may rise freely
# but may not drop. Floors sit a few points under the measured baseline so
# runner noise cannot flip the gate red.
#
# Measured 2026-10 (must stay in sync with .github/workflows/ci.yml):
#   engine 80.3, device 75.3
RATCHET_PKGS=(
    "./internal/engine/...:80.0"
    "./internal/device/...:75.0"
)

# package_coverage prints the coverage percentage for the package matching
# the given pattern, or nothing if the package is not in the profile.
package_coverage() {
    local pattern="$1"
    # Translate a "./internal/foo/..." pattern into the module path prefix.
    local prefix="${pattern#./}"
    prefix="${prefix%/...}"
    prefix="${prefix//\//\/}"

    awk -v prefix="longbow-quarrel/${prefix}" '
        NR == 1 && $0 ~ /^mode:/ { next }
        {
            path = $1
            sub(/:[0-9]+[.,][0-9]+$/, "", path)

            # Only count files under the requested package directory.
            idx = index(path, prefix "/")
            if (idx == 0) next
            rest = substr(path, idx + length(prefix) + 1)
            if (rest == "" || rest ~ /\//) next

            total += $2
            if ($3 > 0) covered += $2
        }
        END {
            if (total > 0) printf "%.1f\n", 100.0 * covered / total
        }
    ' "${PROFILE}"
}

FAILED=0

echo "=== Coverage gate (>= 80% on core packages) ==="
for pkg in "${CORE_PKGS[@]}"; do
    COV=$(package_coverage "${pkg}")
    if [[ -z "${COV}" ]]; then
        echo "::error::${pkg} produced no coverage result from ${PROFILE}"
        FAILED=1
        continue
    fi
    echo "Package ${pkg}: ${COV}%"
    if [[ "$(awk -v c="${COV}" 'BEGIN { print (c < 80.0) }')" == "1" ]]; then
        echo "::error::Package ${pkg} coverage (${COV}%) is below the required 80% threshold!"
        FAILED=1
    fi
done

echo ""
echo "=== Coverage ratchet floors ==="
for entry in "${RATCHET_PKGS[@]}"; do
    pkg="${entry%:*}"
    floor="${entry##*:}"
    COV=$(package_coverage "${pkg}")
    if [[ -z "${COV}" ]]; then
        echo "::error::${pkg} produced no coverage result from ${PROFILE}"
        FAILED=1
        continue
    fi
    if [[ "$(awk -v c="${COV}" -v f="${floor}" 'BEGIN { print (c < f) }')" == "1" ]]; then
        echo "::error::Package ${pkg} coverage (${COV}%) fell below its floor (${floor}%)"
        FAILED=1
    else
        echo "Package ${pkg}: ${COV}% (floor ${floor}%)"
    fi
done

echo ""
if [[ ${FAILED} -ne 0 ]]; then
    echo "coverage_gate: FAILED"
    exit 1
fi

echo "coverage_gate: OK - all coverage gates satisfied"
exit 0
