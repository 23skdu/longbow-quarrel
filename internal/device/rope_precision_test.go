//go:build darwin && metal

package device

import (
	"fmt"
	"math"
	"testing"
)

func TestRoPE_ThetaCalculation_Large(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 128
	pos := 5
	ropeTheta := float32(1000000.0)

	testIdx := 32

	thetaCPU := float64(pos) * math.Pow(float64(ropeTheta), -2.0*float64(testIdx)/float64(headDim))
	fmt.Printf("CPU theta for idx=%d, pos=%d, theta=%.0f: %.10f\n", testIdx, pos, ropeTheta, thetaCPU)

	inputData := make([]float32, 256)
	inputData[testIdx] = 1.0
	inputData[testIdx+headDim/2] = 2.0

	tensor := ctx.NewTensor(1, 256)
	tensor.LoadFrom(inputData)
	defer tensor.ReturnToPool()

	tensor.RoPE(pos, headDim, 1, ropeTheta)
	ctx.Synchronize()

	result := tensor.ToHost()

	thetaGPUExpected := math.Pow(float64(ropeTheta), -2.0*float64(testIdx)/float64(headDim))
	fmt.Printf("Expected theta = %.10f\n", thetaGPUExpected)

	fmt.Printf("GPU: idx0=%d -> %.6f, idx%d -> %.6f\n", testIdx, result[testIdx], testIdx+headDim/2, result[testIdx+headDim/2])

	theta_i := (float)pos * pow(ropeTheta, -2.0f * (float)testIdx / (float)headDim)

	fmt.Printf("Kernel theta_i = %.10f\n", theta_i)

	sinTheta := math.Sin(thetaCPU)
	cosTheta := math.Cos(thetaCPU)

	expected0 := 1.0 * float32(cosTheta) - 2.0 * float32(sinTheta)
	expected1 := 1.0 * float32(sinTheta) + 2.0 * float32(cosTheta)

	fmt.Printf("Expected: idx0=%.6f, idx1=%.6f\n", expected0, expected1)

	if math.Abs(float64(result[testIdx]-expected0)) > 1e-3 {
		t.Errorf("idx0 mismatch: got %.6f, want %.6f", result[testIdx], expected0)
	}
	if math.Abs(float64(result[testIdx+headDim/2]-expected1)) > 1e-3 {
		t.Errorf("idx1 mismatch: got %.6f, want %.6f", result[testIdx+headDim/2], expected1)
	}
}

func TestRoPE_Precision_1M_Theta(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 128
	heads := 32
	pos := 10
	ropeTheta := float32(1000000.0)

	inputData := make([]float32, heads*headDim)
	for i := range inputData {
		inputData[i] = float32(i+1) / 100.0
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

	t.Logf("Mistral 1M theta (ropeTheta=%.0f) max diff from CPU: %.6f", ropeTheta, maxDiff)

	if maxDiff > 0.001 {
		t.Errorf("Mistral 1M theta RoPE has unacceptable deviation from CPU: %.6f > 0.001", maxDiff)
	}

	if maxDiff > 0.01 {
		t.Errorf("Mistral 1M theta RoPE has severe deviation from CPU: %.6f", maxDiff)
	}
}
