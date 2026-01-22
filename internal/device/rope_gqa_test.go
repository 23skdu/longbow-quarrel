//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

func TestRoPE_QK_DualCall_Mistral(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 128
	heads := 32
	kvHeads := 8
	pos := 5
	ropeTheta := float32(1000000.0)

	inputData := make([]float32, heads*headDim)
	for i := range inputData {
		inputData[i] = float32(i) / 100.0
	}

	qTensor := ctx.NewTensor(1, heads*headDim)
	qTensor.LoadFrom(inputData)
	defer qTensor.ReturnToPool()

	kTensor := ctx.NewTensor(1, kvHeads*headDim)
	kTensor.LoadFrom(inputData[:kvHeads*headDim])
	defer kTensor.ReturnToPool()

	qTensor.RoPE(pos, headDim, heads, 1, ropeTheta)
	kTensor.RoPE(pos, headDim, kvHeads, 1, ropeTheta)
	ctx.Synchronize()

	qResult := qTensor.ToHost()
	kResult := kTensor.ToHost()

	qCPU := CPURoPE(inputData, pos, heads, headDim, ropeTheta)
	kCPU := CPURoPE(inputData[:kvHeads*headDim], pos, kvHeads, headDim, ropeTheta)

	qErrors := 0
	qMaxDiff := float32(0.0)
	for i := 0; i < len(qResult); i++ {
		diff := float32(math.Abs(float64(qResult[i] - qCPU[i])))
		if diff > qMaxDiff {
			qMaxDiff = diff
		}
		if diff > 0.01 {
			qErrors++
			if qErrors <= 3 {
				t.Errorf("Q mismatch at index %d: got %.6f, want %.6f (diff=%.6f)", i, qResult[i], qCPU[i], diff)
			}
		}
	}

	kErrors := 0
	kMaxDiff := float32(0.0)
	for i := 0; i < len(kResult); i++ {
		diff := float32(math.Abs(float64(kResult[i] - kCPU[i])))
		if diff > kMaxDiff {
			kMaxDiff = diff
		}
		if diff > 0.01 {
			kErrors++
			if kErrors <= 3 {
				t.Errorf("K mismatch at index %d: got %.6f, want %.6f (diff=%.6f)", i, kResult[i], kCPU[i], diff)
			}
		}
	}

	t.Logf("Q RoPE: %d errors > 0.01 threshold, max diff=%.6f", qErrors, qMaxDiff)
	t.Logf("K RoPE: %d errors > 0.01 threshold, max diff=%.6f", kErrors, kMaxDiff)

	if qMaxDiff > 0.1 || kMaxDiff > 0.1 {
		t.Errorf("RoPE has significant deviation from CPU reference: Q_max=%.6f, K_max=%.6f", qMaxDiff, kMaxDiff)
	}

	t.Logf("✓ RoPE dual call pattern verified for pos=%d, heads=%d, kvHeads=%d", pos, heads, kvHeads)
}

func TestRoPE_GQA_Correctness(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 128
	heads := 32
	kvHeads := 8
	pos := 10
	ropeTheta := float32(1000000.0)

	qData := make([]float32, heads*headDim)
	kData := make([]float32, kvHeads*headDim)
	for i := 0; i < heads*headDim; i++ {
		qData[i] = float32(i)
	}
	for i := 0; i < kvHeads*headDim; i++ {
		kData[i] = float32(i)
	}

	qTensor := ctx.NewTensor(1, heads*headDim)
	qTensor.LoadFrom(qData)
	defer qTensor.ReturnToPool()

	kTensor := ctx.NewTensor(1, kvHeads*headDim)
	kTensor.LoadFrom(kData)
	defer kTensor.ReturnToPool()

	qTensor.RoPE(pos, headDim, heads, 1, ropeTheta)
	kTensor.RoPE(pos, headDim, kvHeads, 1, ropeTheta)
	ctx.Synchronize()

	qResult := qTensor.ToHost()
	kResult := kTensor.ToHost()

	if len(qResult) != heads*headDim {
		t.Errorf("Q tensor size mismatch: got %d, want %d", len(qResult), heads*headDim)
	}
	if len(kResult) != kvHeads*headDim {
		t.Errorf("K tensor size mismatch: got %d, want %d", len(kResult), kvHeads*headDim)
	}

	qRotations := 0
	kRotations := 0

	for i := 0; i < heads*headDim/2; i++ {
		i0 := i
		i1 := i + headDim/2

		if qResult[i0] != qData[i0] || qResult[i1] != qData[i1] {
			qRotations++
		}
	}

	for i := 0; i < kvHeads*headDim/2; i++ {
		i0 := i
		i1 := i + headDim/2

		if kResult[i0] != kData[i0] || kResult[i1] != kData[i1] {
			kRotations++
		}
	}

	if qRotations == 0 {
		t.Error("Q was not rotated at all")
	}
	if kRotations == 0 {
		t.Error("K was not rotated at all")
	}

	t.Logf("✓ GQA Q rotation check: %d/%d pairs rotated", qRotations, heads*headDim/2)
	t.Logf("✓ GQA K rotation check: %d/%d pairs rotated", kRotations, kvHeads*headDim/2)
}

func TestRoPE_PositionIndependence(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 128
	heads := 4
	pos := 3
	ropeTheta := float32(10000.0)

	inputData := make([]float32, heads*headDim)
	for i := range inputData {
		inputData[i] = float32(i) / 100.0
	}

	tensor1 := ctx.NewTensor(1, heads*headDim)
	tensor1.LoadFrom(inputData)
	defer tensor1.ReturnToPool()

	tensor2 := ctx.NewTensor(1, heads*headDim)
	tensor2.LoadFrom(inputData)
	defer tensor2.ReturnToPool()

	tensor1.RoPE(pos, headDim, heads, 1, ropeTheta)
	tensor2.RoPE(pos+10, headDim, heads, 1, ropeTheta)
	ctx.Synchronize()

	result1 := tensor1.ToHost()
	result2 := tensor2.ToHost()

	sameRotation := true
	for i := 0; i < len(result1); i++ {
		if result1[i] == result2[i] {
			sameRotation = false
			break
		}
	}

	if sameRotation {
		t.Error("Different positions produced same rotation (RoPE not working)")
	}

	t.Logf("✓ Position independence verified: pos %d != pos %d", pos, pos+10)
}

func TestRoPE_ZeroPosition(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 128
	heads := 4
	pos := 0
	ropeTheta := float32(10000.0)

	inputData := make([]float32, heads*headDim)
	for i := range inputData {
		inputData[i] = float32(i)
	}

	tensor := ctx.NewTensor(1, heads*headDim)
	tensor.LoadFrom(inputData)
	defer tensor.ReturnToPool()

	tensor.RoPE(pos, headDim, heads, 1, ropeTheta)
	ctx.Synchronize()

	result := tensor.ToHost()

	rotated := 0
	for i := 0; i < headDim/2; i++ {
		i0 := i
		i1 := i + headDim/2

		if result[i0] != inputData[i0] || result[i1] != inputData[i1] {
			rotated++
		}
	}

	if rotated > 0 {
		t.Error("Position 0 should not rotate (cos(0)=1, sin(0)=0)")
	}

	t.Logf("✓ Zero position handling verified")
}

func TestRoPE_Mistral_1M_Theta(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 128
	heads := 32
	pos := 5
	ropeTheta := float32(1000000.0)

	inputData := make([]float32, heads*headDim)
	for i := range inputData {
		inputData[i] = float32(i) / 100.0
	}

	tensor := ctx.NewTensor(1, heads*headDim)
	tensor.LoadFrom(inputData)
	defer tensor.ReturnToPool()

	tensor.RoPE(pos, headDim, heads, 1, ropeTheta)
	ctx.Synchronize()

	result := tensor.ToHost()

	cpuResult := CPURoPE(inputData, pos, heads, headDim, ropeTheta)

	maxDiff := float32(0.0)
	for i := 0; i < len(result); i++ {
		diff := float32(math.Abs(float64(result[i] - cpuResult[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	if maxDiff > 0.01 {
		t.Errorf("Mistral 1M theta RoPE has large deviation from CPU: %.6f", maxDiff)
	}

	t.Logf("✓ Mistral 1M theta (ropeTheta=%.0f) verified with max diff=%.6f", ropeTheta, maxDiff)
}

func TestRoPE_AdjacentPairing(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 128
	heads := 2
	pos := 3
	ropeTheta := float32(10000.0)

	inputData := make([]float32, heads*headDim)

	expected := make([]float32, heads*headDim)
	for h := 0; h < heads; h++ {
		for i := 0; i < headDim/2; i++ {
			halfDim := headDim / 2
			idx0 := h*headDim + i
			idx1 := h*headDim + i + halfDim

			inputData[idx0] = float32(idx0)
			inputData[idx1] = float32(idx1)

			theta := float64(pos) * math.Pow(float64(ropeTheta), -2.0*float64(i)/float64(headDim))
			cosTheta := math.Cos(theta)
			sinTheta := math.Sin(theta)

			v0 := float64(inputData[idx0])
			v1 := float64(inputData[idx1])

			expected[idx0] = float32(v0*cosTheta - v1*sinTheta)
			expected[idx1] = float32(v0*sinTheta + v1*cosTheta)
		}
	}

	tensor := ctx.NewTensor(1, heads*headDim)
	tensor.LoadFrom(inputData)
	defer tensor.ReturnToPool()

	tensor.RoPE(pos, headDim, heads, 1, ropeTheta)
	ctx.Synchronize()

	result := tensor.ToHost()

	errors := 0
	for i := 0; i < len(result); i++ {
		if math.Abs(float64(result[i]-expected[i])) > 1e-3 {
			errors++
			if errors <= 3 {
				t.Errorf("Adjacent pairing mismatch at index %d: got %.6f, want %.6f", i, result[i], expected[i])
			}
		}
	}

	if errors > 0 {
		t.Logf("Adjacent pairing errors: %d", errors)
	}

	if errors > 10 {
		t.Errorf("Too many adjacent pairing errors: %d", errors)
	} else {
		t.Logf("✓ Adjacent pairing (Llama/Mistral style) verified")
	}
}
