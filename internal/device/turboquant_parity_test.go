package device

import (
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/simd"
)

// turboQuantReferenceEncode mirrors the CUDA kernel
// turboquant_encode_kernel in internal/device/cuda_kernels.cu.
//
// The CPU encoder must produce the same bytes as the GPU encoder, otherwise
// the same model produces a different KV cache depending on the backend.
//
// The CUDA kernel:
//   - rotates:      rotated = R * x
//   - scales:       scale = max|rotated| / maxQuantVal
//   - quantizes:    q = clamp(round(rotated / scale), +/-maxQuantVal)
//   - residuals:    resRotated = rotated - q*scale
//   - rotates back: residual = R^T * resRotated      <- original space
//   - QJL:          qj = sign(qjlMatrix * residual)
//
// The previous x86 implementation returned the residual in rotated space and
// did not clamp, so the CPU and GPU KV caches diverged.
func turboQuantReferenceEncode(input, rotation, qjlMatrix []float32, n, qjlRows, bits int) ([]int8, []int8, float32) {
	maxQuantInt := (1 << uint(bits-1)) - 1
	maxQuantVal := float32(maxQuantInt)

	rotated := make([]float32, n)
	maxAbs := float32(0)
	for i := 0; i < n; i++ {
		var sum float32
		for j := 0; j < n; j++ {
			sum += rotation[i*n+j] * input[j]
		}
		rotated[i] = sum
		if a := float32(math.Abs(float64(sum))); a > maxAbs {
			maxAbs = a
		}
	}
	scale := float32(1)
	if maxAbs > 0 {
		scale = maxAbs / maxQuantVal
	}

	q := make([]int8, n)
	resRotated := make([]float32, n)
	for i := 0; i < n; i++ {
		qf := float32(math.Round(float64(rotated[i] * (1.0 / scale))))
		if qf > maxQuantVal {
			qf = maxQuantVal
		} else if qf < -maxQuantVal {
			qf = -maxQuantVal
		}
		q[i] = int8(qf)
		resRotated[i] = rotated[i] - qf*scale
	}

	residual := make([]float32, n)
	for i := 0; i < n; i++ {
		var sum float32
		for j := 0; j < n; j++ {
			sum += rotation[j*n+i] * resRotated[j]
		}
		residual[i] = sum
	}

	qj := make([]int8, qjlRows)
	for i := 0; i < qjlRows; i++ {
		var sum float32
		for j := 0; j < n; j++ {
			sum += qjlMatrix[i*n+j] * residual[j]
		}
		if sum >= 0 {
			qj[i] = 1
		} else {
			qj[i] = -1
		}
	}

	return q, qj, scale
}

// TestTurboQuantEncode_CPUMatchesCUDASemantics checks the CPU encoder
// against a transcription of the CUDA kernel.
//
// This is the CPU/GPU parity guard for the TurboQuant KV cache path.
func TestTurboQuantEncode_CPUMatchesCUDASemantics(t *testing.T) {
	const n = 64
	const qjlRows = 16
	const bits = 4

	devCtx := NewContext()
	defer devCtx.Free()

	rotation := devCtx.NewTensorFP32(n, n)
	qjl := devCtx.NewTensorFP32(qjlRows, n)
	if rotation == nil || qjl == nil {
		t.Fatal("failed to create the rotation tensors")
	}
	defer rotation.Free()
	defer qjl.Free()

	rotData := make([]float32, n*n)
	qjlData := make([]float32, qjlRows*n)
	for i := range rotData {
		rotData[i] = float32((i%17)-8) / 8.0
	}
	for i := range qjlData {
		if i%3 == 0 {
			qjlData[i] = -1
		} else {
			qjlData[i] = 1
		}
	}
	if err := rotation.LoadFrom(rotData); err != nil {
		t.Fatalf("failed to load the rotation matrix: %v", err)
	}
	if err := qjl.LoadFrom(qjlData); err != nil {
		t.Fatalf("failed to load the QJL matrix: %v", err)
	}

	input := devCtx.NewTensorFP32(1, n)
	if input == nil {
		t.Fatal("failed to create the input tensor")
	}
	defer input.Free()
	inData := make([]float32, n)
	for i := range inData {
		inData[i] = float32((i%13)-6) * 0.25
	}
	if err := input.LoadFrom(inData); err != nil {
		t.Fatalf("failed to load the input: %v", err)
	}

	// The CPU encoder, as used by TurboQuantEncode.
	gotQ, gotScale, gotRes := simd.PolarQuantSIMD(inData, rotData, n, bits)
	gotQJ, _ := simd.QJLTransformSIMD(gotRes, qjlData, qjlRows, n)

	wantQ, wantQJ, wantScale := turboQuantReferenceEncode(inData, rotData, qjlData, n, qjlRows, bits)

	if math.Abs(float64(gotScale-wantScale)) > 1e-5*math.Max(1, math.Abs(float64(wantScale))) {
		t.Errorf("scale = %v, CUDA semantics give %v", gotScale, wantScale)
	}
	for i := range wantQ {
		if gotQ[i] != wantQ[i] {
			t.Fatalf("quantized[%d] = %d, CUDA semantics give %d", i, gotQ[i], wantQ[i])
		}
	}
	for i := range wantQJ {
		if gotQJ[i] != wantQJ[i] {
			t.Fatalf("qjl[%d] = %d, CUDA semantics give %d", i, gotQJ[i], wantQJ[i])
		}
	}
}
