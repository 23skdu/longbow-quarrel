#!/usr/bin/env bash
# ==============================================================================
# Longbow-Quarrel Benchmark Regression Gate
# ==============================================================================
# Compares Go benchmarks between two working trees (base vs head) so a latency
# regression fails CI instead of being scrolled past in an artifact.
#
#   bench_gate.sh measure [pkg...] > results.txt
#       Runs the benchmarks and prints "<BenchmarkName> <ns/op>" using the
#       best (minimum) of BENCH_COUNT runs, which is the most stable estimate
#       on a shared CI runner.
#
#   bench_gate.sh compare BASE.txt HEAD.txt [THRESHOLD]
#       Fails when any benchmark present in both files is more than THRESHOLD
#       times slower in HEAD than in BASE. THRESHOLD defaults to 1.5 because
#       shared runners show ~30% run-to-run variance on parallel benchmarks;
#       the gate is there to catch real 2x-class regressions.
# ==============================================================================

set -euo pipefail

BENCH_FILTER="${BENCH_FILTER:-.}"
BENCH_COUNT="${BENCH_COUNT:-5}"
BENCH_BENCHTIME="${BENCH_BENCHTIME:-100ms}"

mode="${1:-}"
shift || true

case "${mode}" in
measure)
    pkgs=("$@")
    if [ "${#pkgs[@]}" -eq 0 ]; then
        pkgs=(./internal/gguf ./internal/simd ./internal/cpu)
    fi
    go test -run '^$' -bench "${BENCH_FILTER}" -benchtime "${BENCH_BENCHTIME}" \
        -count "${BENCH_COUNT}" "${pkgs[@]}" 2>&1 |
        awk '
            /^Benchmark/ {
                name = $1
                sub(/-[0-9]+$/, "", name)
                for (i = 1; i < NF; i++) {
                    if ($(i + 1) == "ns/op") {
                        v = $i + 0
                        if (!(name in best) || v < best[name]) best[name] = v
                        break
                    }
                }
            }
            END {
                for (name in best) printf "%s %.0f\n", name, best[name]
            }
        ' | sort
    ;;
compare)
    base_file="${1:?usage: bench_gate.sh compare BASE HEAD [THRESHOLD]}"
    head_file="${2:?usage: bench_gate.sh compare BASE HEAD [THRESHOLD]}"
    threshold="${3:-1.5}"

    awk -v thr="${threshold}" '
        NR == FNR { base[$1] = $2; next }
        ($1 in base) {
            seen[$1] = 1
            compared++
            ratio = (base[$1] > 0) ? $2 / base[$1] : 1
            if (ratio > thr) {
                printf "REGRESSION %-40s %10.0f ns/op vs %10.0f ns/op (%.2fx)\n", $1, $2, base[$1], ratio
                bad++
            } else {
                printf "ok         %-40s %10.0f ns/op vs %10.0f ns/op (%.2fx)\n", $1, $2, base[$1], ratio
            }
        }
        END {
            for (n in base) if (!(n in seen)) printf "missing    %-40s (baseline only)\n", n
            if (compared == 0) {
                print "\nbenchmark gate: no overlapping benchmarks between base and head"
                exit 2
            }
            if (bad > 0) {
                printf "\nbenchmark gate: %d benchmark(s) slower than %.2fx of baseline\n", bad, thr
                exit 1
            }
            printf "\nbenchmark gate: %d benchmark(s) within %.2fx of baseline\n", compared, thr
        }
    ' "${base_file}" "${head_file}"
    ;;
*)
    echo "usage: bench_gate.sh measure [pkg...] | compare BASE.txt HEAD.txt [THRESHOLD]" >&2
    exit 64
    ;;
esac
