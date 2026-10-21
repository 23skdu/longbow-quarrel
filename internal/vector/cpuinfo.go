package vector

import (
	"os"
	"sync"

	"github.com/23skdu/longbow-quarrel/internal/simd"
)

// CPU level constants for the vector kernels. They mirror the levels of
// internal/simd so the two packages report consistently.
const (
	VecLevelScalar = 0
	VecLevelAVX2   = 1
	VecLevelAVX512 = 2
)

// Kernel availability flags. The *_compiled flags are set by init() in
// the build-tagged files that carry the cgo kernels; without the right
// tag the C objects do not exist and the dispatcher must not touch the
// function pointers.
var (
	kernelsCompiledAVX2   bool
	kernelsCompiledAVX512 bool
)

var (
	vecCPUOnce sync.Once

	vecHasAVX2   bool
	vecHasAVX512 bool

	// testingForceLevel >= 0 pins the dispatch level so the scalar
	// reference and each kernel set can be exercised from tests. -1
	// means "detect".
	testingForceLevel = -1
)

var (
	vecEnvDisableSIMD   = "DISABLE_SIMD"
	vecEnvDisableAVX2   = "DISABLE_AVX2"
	vecEnvDisableAVX512 = "DISABLE_AVX512"
)

func vecDetectCPU() {
	vecCPUOnce.Do(func() {
		vecHasAVX2 = simd.HasAVX2()
		vecHasAVX512 = simd.HasAVX512()

		// The same overrides internal/simd honours, so a test that
		// wants the scalar path of every package can set one
		// environment variable.
		if os.Getenv(vecEnvDisableSIMD) == "1" {
			vecHasAVX2 = false
			vecHasAVX512 = false
		}
		if os.Getenv(vecEnvDisableAVX512) == "1" {
			vecHasAVX512 = false
		}
		if os.Getenv(vecEnvDisableAVX2) == "1" {
			vecHasAVX2 = false
			vecHasAVX512 = false
		}
	})
}

// vecLevel reports the effective capability level. A level is only
// usable when the kernels for it were compiled in with a build tag.
func vecLevel() int {
	vecDetectCPU()
	level := VecLevelScalar
	switch {
	case vecHasAVX512 && kernelsCompiledAVX512:
		level = VecLevelAVX512
	case vecHasAVX2 && (kernelsCompiledAVX512 || kernelsCompiledAVX2):
		// kernelsCompiledAVX512 implies the AVX2 kernels were compiled
		// with it (see distance_avx512.go, which wires both sets, the
		// same way internal/simd/avx512.go declares the avx2 kernels).
		level = VecLevelAVX2
	}
	if testingForceLevel >= 0 && testingForceLevel < level {
		return testingForceLevel
	}
	return level
}

// resetVectorCPUForTesting re-runs the capability probe; the forced
// level applies until a test changes it back.
func resetVectorCPUForTesting(force int) {
	vecCPUOnce = sync.Once{}
	testingForceLevel = force
}
