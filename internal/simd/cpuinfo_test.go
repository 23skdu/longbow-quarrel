//go:build !avx512

package simd

import (
	"os"
	"strings"
	"testing"
)

func TestCPUDetection(t *testing.T) {
	// Reset and re-detect
	cpuInitDone = false
	detectCPU()

	if !cpuInitDone {
		t.Fatal("detectCPU() should mark detection as done")
	}

	if GetCPULevel() < CPULevelAVX2 {
		t.Logf("CPU level: %d (AVX2 not detected, running on non-x86?)", GetCPULevel())
	}
}

func TestContainsFlag(t *testing.T) {
	tests := []struct {
		flags string
		flag  string
		want  bool
	}{
		{"avx2 avx512f fma", "avx2", true},
		{"avx2 avx512f fma", "avx512f", true},
		{"avx2 avx512f fma", "avx3", false},
		{"avx2 avx_vnni fma", "avx_vnni", true},
		{"avx2", "avx2", true},
		{"", "avx2", false},
	}

	for _, tt := range tests {
		if got := containsFlag(tt.flags, tt.flag); got != tt.want {
			t.Errorf("containsFlag(%q, %q) = %v, want %v", tt.flags, tt.flag, got, tt.want)
		}
	}
}

func TestReadCPUFlags(t *testing.T) {
	flags := readCPUFlags()
	if flags == "" {
		t.Skip("not on Linux or /proc/cpuinfo not available")
	}
	if !strings.Contains(flags, "avx2") && !strings.Contains(flags, "asimd") {
		t.Logf("No SIMD flags found: %s", flags[:min(len(flags), 200)])
	}
}

func TestDisableAVX2Env(t *testing.T) {
	os.Setenv("DISABLE_AVX2", "1")
	defer os.Unsetenv("DISABLE_AVX2")

	cpuInitDone = false
	detectCPU()

	if hasAVX2 {
		t.Error("hasAVX2 should be false when DISABLE_AVX2=1")
	}
	if hasAVX512 {
		t.Error("hasAVX512 should be false when DISABLE_AVX2=1")
	}
}

func TestDisableAVX512Env(t *testing.T) {
	os.Setenv("DISABLE_AVX512", "1")
	defer os.Unsetenv("DISABLE_AVX512")

	cpuInitDone = false
	detectCPU()

	if hasAVX512 {
		t.Error("hasAVX512 should be false when DISABLE_AVX512=1")
	}
}

func TestUseAVX2(t *testing.T) {
	detectCPU()
	if hasAVX2 {
		if !useAVX2() {
			t.Error("useAVX2() should return true when hasAVX2 is true")
		}
	} else {
		if useAVX2() {
			t.Error("useAVX2() should return false when hasAVX2 is false")
		}
	}
}
