package simd

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/mathx"
)

func TestAVX2Scalar_Coverage(t *testing.T) {
	// Empty slice test for softmaxScalar
	softmaxScalar([]float32{})

	// Single element test for softmaxScalar
	x := []float32{1.0}
	softmaxScalar(x)
	if x[0] != 1.0 {
		t.Errorf("expected 1.0, got %f", x[0])
	}
}

func TestCPUInfo_Coverage(t *testing.T) {
	origOS := cpuInfoOS
	origPath := cpuInfoPath
	origExec := execSysctl
	origCustom := readCPUFlagsCustom
	defer func() {
		cpuInfoOS = origOS
		cpuInfoPath = origPath
		execSysctl = origExec
		readCPUFlagsCustom = origCustom
		resetCPUInitForTesting()
		detectCPU()
	}()

	// 1. DISABLE_SIMD
	t.Setenv("DISABLE_SIMD", "1")
	resetCPUInitForTesting()
	detectCPU()
	if GetCPULevel() != CPULevelScalar {
		t.Errorf("expected CPULevelScalar with DISABLE_SIMD=1, got %d", GetCPULevel())
	}
	t.Setenv("DISABLE_SIMD", "")

	// 2. Flags with AVX512, AMX, AVX-VNNI
	readCPUFlagsCustom = func() string {
		return "avx2 avx512f avx512bw amx_tile amx_int8 avx_vnni"
	}
	resetCPUInitForTesting()
	detectCPU()
	if GetCPULevel() != CPULevelAVX512 {
		t.Errorf("expected CPULevelAVX512, got %d", GetCPULevel())
	}
	if !HasAMX() {
		t.Error("expected HasAMX to be true")
	}
	if !HasAVXVNNI() {
		t.Error("expected HasAVXVNNI to be true")
	}

	// 3. DISABLE_AVX512
	t.Setenv("DISABLE_AVX512", "1")
	resetCPUInitForTesting()
	detectCPU()
	if GetCPULevel() != CPULevelAVX2 {
		t.Errorf("expected CPULevelAVX2 with DISABLE_AVX512=1, got %d", GetCPULevel())
	}
	t.Setenv("DISABLE_AVX512", "")

	// 4. DISABLE_AVX2
	t.Setenv("DISABLE_AVX2", "1")
	resetCPUInitForTesting()
	detectCPU()
	if GetCPULevel() != CPULevelScalar {
		t.Errorf("expected CPULevelScalar with DISABLE_AVX2=1, got %d", GetCPULevel())
	}
	t.Setenv("DISABLE_AVX2", "")

	// 5. Calling getters when !cpuInitDone
	readCPUFlagsCustom = func() string { return "avx2 amx_tile amx_bf16 avx512_vnni" }
	resetCPUInitForTesting()
	if GetCPULevel() != CPULevelAVX2 {
		t.Errorf("expected CPULevelAVX2 from GetCPULevel(), got %d", GetCPULevel())
	}

	resetCPUInitForTesting()
	if !HasAMX() {
		t.Error("expected HasAMX() to initialize and return true")
	}

	resetCPUInitForTesting()
	if !HasAVXVNNI() {
		t.Error("expected HasAVXVNNI() to initialize and return true")
	}

	readCPUFlagsCustom = nil

	// 6. readCPUFlags OS switch
	cpuInfoOS = "windows"
	if flags := readCPUFlags(); flags != "" {
		t.Errorf("expected empty flags on windows, got %q", flags)
	}

	// 7. readCPUFlagsLinux with non-existent file
	cpuInfoOS = "linux"
	cpuInfoPath = "/nonexistent/proc/cpuinfo"
	if flags := readCPUFlags(); flags != "" {
		t.Errorf("expected empty flags on missing file, got %q", flags)
	}

	// 8. readCPUFlagsLinux with file without flags
	tmpDir := t.TempDir()
	emptyCpuinfo := filepath.Join(tmpDir, "cpuinfo")
	_ = os.WriteFile(emptyCpuinfo, []byte("processor: 0\nmodel name: MockCPU\n"), 0o600)
	cpuInfoPath = emptyCpuinfo
	if flags := readCPUFlags(); flags != "" {
		t.Errorf("expected empty flags on cpuinfo without flags, got %q", flags)
	}

	// 9. readCPUFlagsDarwin
	cpuInfoOS = "darwin"
	// 9a. Both sysctl commands fail
	execSysctl = func(name string, args ...string) *exec.Cmd {
		return exec.Command("false")
	}
	if flags := readCPUFlags(); flags != "" {
		t.Errorf("expected empty flags when sysctl fails, got %q", flags)
	}

	// 9b. First sysctl fails, second succeeds
	execSysctl = func(name string, args ...string) *exec.Cmd {
		if len(args) > 0 && args[0] == "-a" {
			return exec.Command("false")
		}
		return exec.Command("echo", "hw.optional.avx2_0: 1\nhw.optional.fma: 1")
	}
	flags := readCPUFlags()
	if !containsFlag(flags, "avx2") || !containsFlag(flags, "fma") {
		t.Errorf("expected avx2 and fma flags from fallback sysctl, got %q", flags)
	}

	// 9c. First sysctl succeeds
	execSysctl = func(name string, args ...string) *exec.Cmd {
		return exec.Command("echo", "hw.optional.avx2_0: 1")
	}
	if flags := readCPUFlags(); !containsFlag(flags, "avx2") {
		t.Errorf("expected avx2 from primary sysctl, got %q", flags)
	}
}

func TestSimd_EdgeCases(t *testing.T) {
	// 1. SoftmaxF32 empty
	SoftmaxF32([]float32{})

	// 2. VecDotF32 empty & mismatched lengths
	if res := VecDotF32([]float32{}, []float32{}); res != 0 {
		t.Errorf("expected 0 from empty VecDotF32, got %f", res)
	}
	if res := VecDotF32([]float32{1.0, 2.0}, []float32{3.0}); res != 3.0 {
		t.Errorf("expected 3.0 from mismatched lengths VecDotF32, got %f", res)
	}

	// 3. VecFMAF32 empty & mismatched lengths
	dst := []float32{}
	VecFMAF32(dst, []float32{}, 1.0)
	dstShort := []float32{1.0, 2.0}
	VecFMAF32(dstShort, []float32{3.0}, 2.0)
	if dstShort[0] != 7.0 || dstShort[1] != 2.0 {
		t.Errorf("unexpected VecFMAF32 with mismatched lengths: %v", dstShort)
	}

	// 4. MatMul edge cases
	if res := MatMul(nil, nil, 0, 0, 0); len(res) != 0 {
		t.Errorf("expected empty result from MatMul with zeros, got len %d", len(res))
	}

	// MatMul small (rowsA*colsB < 1024)
	aSmall := []float32{1, 2, 3, 4}
	bSmall := []float32{5, 6, 7, 8}
	resSmall := MatMul(aSmall, bSmall, 2, 2, 2)
	if len(resSmall) != 4 || resSmall[0] != 19 || resSmall[1] != 22 {
		t.Errorf("unexpected MatMul small result: %v", resSmall)
	}

	// MatMul large (rowsA*colsB >= 1024) with GOMAXPROCS(32) for partition branches
	origProcs := runtime.GOMAXPROCS(32)
	defer runtime.GOMAXPROCS(origProcs)

	rowsA := 33
	colsA := 16
	colsB := 64
	aLarge := make([]float32, rowsA*colsA)
	bLarge := make([]float32, colsA*colsB)
	for i := range aLarge {
		aLarge[i] = 1.0
	}
	for i := range bLarge {
		bLarge[i] = 1.0
	}
	resLarge := MatMul(aLarge, bLarge, rowsA, colsA, colsB)
	if len(resLarge) != rowsA*colsB || resLarge[0] != float32(colsA) {
		t.Errorf("unexpected MatMul large result[0]: %f", resLarge[0])
	}

	// 5. MatVecMul edge cases
	if res := MatVecMul(nil, nil, 0, 0); len(res) != 0 {
		t.Errorf("expected empty from MatVecMul zeros, got len %d", len(res))
	}
	if res := MatVecMul([]float32{1}, []float32{}, 1, 1); len(res) != 1 {
		t.Errorf("expected result from short vector in MatVecMul")
	}

	// MatVecMul large (rows >= 32) with GOMAXPROCS(32) for partition branches
	rowsMVM := 33
	colsMVM := 16
	matMVM := make([]float32, rowsMVM*colsMVM)
	vecMVM := make([]float32, colsMVM)
	for i := range matMVM {
		matMVM[i] = 1.0
	}
	for i := range vecMVM {
		vecMVM[i] = 2.0
	}
	resMVM := MatVecMul(matMVM, vecMVM, rowsMVM, colsMVM)
	if len(resMVM) != rowsMVM || resMVM[0] != float32(colsMVM*2) {
		t.Errorf("unexpected MatVecMul result[0]: %f", resMVM[0])
	}

	// 6. Float16ToFloat32 all branches
	// Zero: +0 (0x0000), -0 (0x8000)
	if f := Float16ToFloat32(0x0000); f != 0.0 {
		t.Errorf("expected +0.0, got %f", f)
	}
	if f := Float16ToFloat32(0x8000); f != 0.0 || mathx.Float32bits(f) != 0x80000000 {
		t.Errorf("expected -0.0, got %f", f)
	}
	// Subnormal: +subnormal (0x0001), -subnormal (0x8001)
	if f := Float16ToFloat32(0x0001); f <= 0 {
		t.Errorf("expected positive subnormal, got %f", f)
	}
	if f := Float16ToFloat32(0x8001); f >= 0 {
		t.Errorf("expected negative subnormal, got %f", f)
	}
	// Infinity: +Inf (0x7C00), -Inf (0xFC00)
	if f := Float16ToFloat32(0x7C00); !mathx.IsInf(float64(f), 1) {
		t.Errorf("expected +Inf, got %f", f)
	}
	if f := Float16ToFloat32(0xFC00); !mathx.IsInf(float64(f), -1) {
		t.Errorf("expected -Inf, got %f", f)
	}
	// NaN: 0x7E00
	if f := Float16ToFloat32(0x7E00); !mathx.IsNaN(float64(f)) {
		t.Errorf("expected NaN, got %f", f)
	}

	// 7. VecDotQ8_0_VNNI error branches
	if res := VecDotQ8_0_VNNI(nil, nil); res != 0 {
		t.Errorf("expected 0 on empty vector, got %f", res)
	}
	if res := VecDotQ8_0_VNNI(nil, make([]float32, 10)); res != 0 {
		t.Errorf("expected 0 on unaligned vector, got %f", res)
	}
	if res := VecDotQ8_0_VNNI(make([]byte, 10), make([]float32, 32)); res != 0 {
		t.Errorf("expected 0 on short data, got %f", res)
	}

	// 8. VecDotQ4_K_VNNI error branches
	if res := VecDotQ4_K_VNNI(nil, nil); res != 0 {
		t.Errorf("expected 0 on empty vector, got %f", res)
	}
	if res := VecDotQ4_K_VNNI(nil, make([]float32, 100)); res != 0 {
		t.Errorf("expected 0 on unaligned vector, got %f", res)
	}
	if res := VecDotQ4_K_VNNI(make([]byte, 10), make([]float32, 256)); res != 0 {
		t.Errorf("expected 0 on short data, got %f", res)
	}

	// 8b. dotQ8_0VNNI & dotQ4KVNNI dispatches
	origDotQ8 := dotQ8_0VNNI
	origDotQ4 := dotQ4KVNNI
	origAVXVNNI := hasAVXVNNI
	origAMXTile := hasAMXTile
	defer func() {
		dotQ8_0VNNI = origDotQ8
		dotQ4KVNNI = origDotQ4
		hasAVXVNNI = origAVXVNNI
		hasAMXTile = origAMXTile
	}()

	hasAVXVNNI = true
	dotQ8_0VNNI = func(data []byte, vector []float32, n int) float32 { return 42.0 }
	dotQ4KVNNI = func(data []byte, vector []float32, n int) float32 { return 84.0 }

	if res := VecDotQ8_0_VNNI(make([]byte, 34), make([]float32, 32)); res != 42.0 {
		t.Errorf("expected 42.0 from dotQ8_0VNNI, got %f", res)
	}
	if res := VecDotQ4_K_VNNI(make([]byte, 144), make([]float32, 256)); res != 84.0 {
		t.Errorf("expected 84.0 from dotQ4KVNNI, got %f", res)
	}

	hasAVXVNNI = false
	hasAMXTile = false
	dotQ8_0VNNI = nil
	dotQ4KVNNI = nil
	if res := VecDotQ8_0_VNNI(make([]byte, 34), make([]float32, 32)); res != 0.0 {
		t.Errorf("expected 0.0 from fallback VecDotQ8_0_VNNI, got %f", res)
	}
	if res := VecDotQ4_K_VNNI(make([]byte, 144), make([]float32, 256)); res != 0.0 {
		t.Errorf("expected 0.0 from fallback VecDotQ4_K_VNNI, got %f", res)
	}

	// 9. MatVecMulQ8_0_VNNI error & parallel branches (rows = 33 for worker loop partitioning)
	if res := MatVecMulQ8_0_VNNI(nil, make([]float32, 10), 4, 10); len(res) != 4 {
		t.Errorf("expected 4 elements from MatVecMulQ8_0_VNNI, got %d", len(res))
	}
	rowsQ8 := 33
	colsQ8 := 32
	dataQ8 := make([]byte, rowsQ8*34)
	vecQ8 := make([]float32, colsQ8)
	resQ8 := MatVecMulQ8_0_VNNI(dataQ8, vecQ8, rowsQ8, colsQ8)
	if len(resQ8) != rowsQ8 {
		t.Errorf("expected %d elements, got %d", rowsQ8, len(resQ8))
	}

	// 10. MatVecMulQ4_K_VNNI error & parallel branches (rows = 33 for worker loop partitioning)
	if res := MatVecMulQ4_K_VNNI(nil, make([]float32, 10), 4, 10); len(res) != 4 {
		t.Errorf("expected 4 elements from MatVecMulQ4_K_VNNI, got %d", len(res))
	}
	rowsQ4 := 33
	colsQ4 := 256
	dataQ4 := make([]byte, rowsQ4*144)
	vecQ4 := make([]float32, colsQ4)
	resQ4 := MatVecMulQ4_K_VNNI(dataQ4, vecQ4, rowsQ4, colsQ4)
	if len(resQ4) != rowsQ4 {
		t.Errorf("expected %d elements, got %d", rowsQ4, len(resQ4))
	}
}

func TestSimdDefault_Coverage(t *testing.T) {
	// 1. SwiGLU length mismatch
	SwiGLU([]float32{1.0}, []float32{}, []float32{})

	// 2. Fp16ToFp32 length mismatch
	Fp16ToFp32([]uint16{1}, []float32{})

	// 3. Fp32ToFp16 length mismatch & underflow
	Fp32ToFp16([]float32{1.0}, []uint16{})

	// Underflow to subnormal: exp > 0 and newExp <= 0 (e.g. 2^-17)
	src := []float32{float32(mathx.Pow(2, -17)), -float32(mathx.Pow(2, -17))}
	dst := make([]uint16, 2)
	Fp32ToFp16(src, dst)
	if dst[0] == 0 || dst[1] == 0 {
		t.Errorf("expected non-zero subnormal float16, got %v", dst)
	}
}

func TestTurboQuant_Coverage(t *testing.T) {
	// 1. PolarQuantSIMD with len(rotationMatrix) == 0
	in := []float32{1.0, -2.0, 3.0}
	q, scale, res := PolarQuantSIMD(in, nil, 3, 4)
	if len(q) != 3 || scale == 0 || len(res) != 3 {
		t.Errorf("unexpected PolarQuantSIMD with nil rotation: %v, %f, %v", q, scale, res)
	}

	// 2. QJLTransformSIMD with negative projection
	signMat := []float32{-1.0, -1.0}
	resid := []float32{2.0, 2.0} // sum will be -4.0 < 0 -> quantized[0] = -1
	qQJL, sQJL := QJLTransformSIMD(resid, signMat, 1, 2)
	if len(qQJL) != 1 || qQJL[0] != -1 || sQJL <= 0 {
		t.Errorf("expected quantized[0] = -1, got %v, scale %f", qQJL, sQJL)
	}
}
