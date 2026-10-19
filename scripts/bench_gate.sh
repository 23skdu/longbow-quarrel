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
#   bench_gate.sh measure-full [pkg...] > results.txt
#       Runs the benchmarks and outputs "<BenchmarkName> <ns/op> <B/op> <allocs/op>".
#
#   bench_gate.sh check-zero-alloc [pkg...]
#       Fails if any hotpath vector/SIMD/decode benchmark reports > 0 allocs/op.
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
measure-full)
    pkgs=("$@")
    if [ "${#pkgs[@]}" -eq 0 ]; then
        pkgs=(./internal/simd ./internal/gguf)
    fi
    go test -run '^$' -bench "${BENCH_FILTER}" -benchtime "${BENCH_BENCHTIME}" \
        -benchmem -count "${BENCH_COUNT}" "${pkgs[@]}" 2>&1 |
        awk '
            /^Benchmark/ {
                name = $1
                sub(/-[0-9]+$/, "", name)
                ns = 0; bop = 0; allocs = 0;
                for (i = 1; i < NF; i++) {
                    if ($(i + 1) == "ns/op") ns = $i + 0
                    if ($(i + 1) == "B/op") bop = $i + 0
                    if ($(i + 1) == "allocs/op") allocs = $i + 0
                }
                if (!(name in best_ns) || ns < best_ns[name]) {
                    best_ns[name] = ns
                    best_bop[name] = bop
                    best_allocs[name] = allocs
                }
            }
            END {
                for (name in best_ns) {
                    printf "%s %.0f %.0f %.0f\n", name, best_ns[name], best_bop[name], best_allocs[name]
                }
            }
        ' | sort
    ;;
check-zero-alloc)
    pkgs=("$@")
    if [ "${#pkgs[@]}" -eq 0 ]; then
        pkgs=(./internal/simd)
    fi
    filter="${BENCH_FILTER}"
    if [ "${filter}" = "." ]; then
        # Default hotpaths that MUST have 0 allocs/op
        filter="Benchmark(VecDotF32$|VecFMAF32$|RMSNorm$|RoPE$|FusedAttention$|FusedMLP$|VecDotQ8_0_VNNI|VecDotQ4_K_VNNI|SwiGLUAVX2$)"
    fi
    echo "Checking zero-allocation gate on hotpath benchmarks with filter: ${filter}..."

    go test -run '^$' -bench "${filter}" -benchtime "50ms" -benchmem -count 1 "${pkgs[@]}" 2>&1 |
        awk '
            /^Benchmark/ {
                name = $1
                sub(/-[0-9]+$/, "", name)
                allocs = -1
                for (i = 1; i < NF; i++) {
                    if ($(i + 1) == "allocs/op") {
                        allocs = $i + 0
                        break
                    }
                }
                if (allocs > 0) {
                    printf "ALLOC_REGRESSION %-45s %d allocs/op (> 0)\n", name, allocs
                    failed++
                } else if (allocs == 0) {
                    printf "ok_zero_alloc    %-45s 0 allocs/op\n", name
                    passed++
                }
            }
            END {
                if (failed > 0) {
                    printf "\nzero-alloc gate FAILED: %d benchmark(s) leaked heap allocations\n", failed
                    exit 1
                }
                if (passed == 0) {
                    printf "\nzero-alloc gate WARNING: no benchmarks matched filter\n"
                    exit 2
                }
                printf "\nzero-alloc gate PASSED: %d hotpath benchmark(s) verified with 0 allocs/op\n", passed
            }
        '
    ;;
compare)
    base_file="${1:?usage: bench_gate.sh compare BASE HEAD [THRESHOLD]}"
    head_file="${2:?usage: bench_gate.sh compare BASE HEAD [THRESHOLD]}"
    threshold="${3:-1.5}"

    awk -v thr="${threshold}" '
        NR == FNR {
            base[$1] = $2
            if (NF >= 4) base_allocs[$1] = $4
            next
        }
        ($1 in base) {
            seen[$1] = 1
            compared++
            ratio = (base[$1] > 0) ? $2 / base[$1] : 1
            alloc_bad = 0
            if (NF >= 4 && ($1 in base_allocs)) {
                if ($4 > base_allocs[$1]) alloc_bad = 1
            }
            if (ratio > thr || alloc_bad) {
                printf "REGRESSION %-40s %10.0f ns/op vs %10.0f ns/op (%.2fx)", $1, $2, base[$1], ratio
                if (alloc_bad) printf " [ALLOC REGRESSION: %d vs %d]", $4, base_allocs[$1]
                printf "\n"
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
                printf "\nbenchmark gate: %d benchmark(s) slower than %.2fx of baseline or regressed allocations\n", bad, thr
                exit 1
            }
            printf "\nbenchmark gate: %d benchmark(s) within %.2fx of baseline\n", compared, thr
        }
    ' "${base_file}" "${head_file}"
    ;;
*)
    echo "usage: bench_gate.sh measure [pkg...] | measure-full [pkg...] | check-zero-alloc [pkg...] | compare BASE.txt HEAD.txt [THRESHOLD]" >&2
    exit 64
    ;;
esac

