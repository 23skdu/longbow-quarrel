//go:build linux && amd64 && cuda && cgo

package device

import (
	"math"
	"testing"
)

func TestCUDA_MOERouterLogits(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// 2 tokens, 4 experts, dim=4
	batchSize := 2
	numExperts := 4
	dim := 4

	input := ctx.NewTensorFP32(batchSize, dim)
	_ = input.LoadFrom([]float32{
		1.0, 0.0, 0.0, 0.0, // Token 0
		0.0, 1.0, 0.0, 0.0, // Token 1
	})
	defer input.Free()

	gateWeight := ctx.NewTensorFP32(numExperts, dim)
	_ = gateWeight.LoadFrom([]float32{
		1.0, 0.0, 0.0, 0.0, // Expert 0: matches Token 0
		0.5, 0.5, 0.0, 0.0, // Expert 1
		0.0, 1.0, 0.0, 0.0, // Expert 2: matches Token 1
		0.0, 0.0, 1.0, 1.0, // Expert 3
	})
	defer gateWeight.Free()

	logits := ctx.MOERouterLogits(input, gateWeight)
	defer logits.Free()

	ctx.Synchronize()
	logitData := logits.ToHost()
	t.Logf("Logits: %v", logitData)

	// Expected:
	// Token 0: [1.0, 0.5, 0.0, 0.0]
	// Token 1: [0.0, 0.5, 1.0, 0.0]
	expected := []float32{
		1.0, 0.5, 0.0, 0.0,
		0.0, 0.5, 1.0, 0.0,
	}
	for i, exp := range expected {
		if math.Abs(float64(logitData[i]-exp)) > 1e-4 {
			t.Errorf("Logit[%d] expected %f, got %f", i, exp, logitData[i])
		}
	}
}

func TestCUDA_MOETopKSelection(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// 2 tokens, 4 experts, top_k=2
	batchSize := 2
	numExperts := 4
	topK := 2

	logits := ctx.NewTensorFP32(batchSize, numExperts)
	_ = logits.LoadFrom([]float32{
		1.0, 0.5, 0.0, 0.0, // Token 0: Top-2 should be Expert 0 and Expert 1
		0.0, 0.5, 1.0, 0.0, // Token 1: Top-2 should be Expert 2 and Expert 1
	})
	defer logits.Free()

	indices, weights := ctx.MOETopKSelection(logits, topK)
	defer indices.Free()
	defer weights.Free()

	ctx.Synchronize()
	idxData := indices.ToHost()
	wData := weights.ToHost()
	t.Logf("Expert Indices: %v", idxData)
	t.Logf("Expert Weights: %v", wData)

	// Token 0: experts 0 and 1
	if int32(idxData[0]) != 0 || int32(idxData[1]) != 1 {
		t.Errorf("Token 0: expected experts [0, 1], got [%v, %v]", idxData[0], idxData[1])
	}
	// Token 1: experts 2 and 1
	if int32(idxData[2]) != 2 || int32(idxData[3]) != 1 {
		t.Errorf("Token 1: expected experts [2, 1], got [%v, %v]", idxData[2], idxData[3])
	}

	// Softmax checks:
	// Token 0: exp(1.0) / (exp(1.0) + exp(0.5)) ≈ 0.622459, exp(0.5) / ... ≈ 0.377541
	exp0 := float32(math.Exp(1.0) / (math.Exp(1.0) + math.Exp(0.5)))
	exp1 := float32(math.Exp(0.5) / (math.Exp(1.0) + math.Exp(0.5)))
	if math.Abs(float64(wData[0]-exp0)) > 1e-4 {
		t.Errorf("Token 0 w[0]: expected %f, got %f", exp0, wData[0])
	}
	if math.Abs(float64(wData[1]-exp1)) > 1e-4 {
		t.Errorf("Token 0 w[1]: expected %f, got %f", exp1, wData[1])
	}
	sum0 := wData[0] + wData[1]
	if math.Abs(float64(sum0-1.0)) > 1e-4 {
		t.Errorf("Token 0 weights sum expected 1.0, got %f", sum0)
	}
}

func TestCUDA_MOEExpertForward(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// 1 token, 2 experts, dim=4, hidden_dim=4, top_k=2
	batchSize := 1
	numExperts := 2
	dim := 4
	hiddenDim := 4
	topK := 2

	input := ctx.NewTensorFP32(batchSize, dim)
	_ = input.LoadFrom([]float32{1, 1, 1, 1})
	defer input.Free()

	// Expert weights: [hidden_dim * num_experts, dim]
	// Expert 0: all 1.0
	// Expert 1: all 2.0
	expertWeights := ctx.NewTensorFP32(hiddenDim*numExperts, dim)
	exWeightsData := make([]float32, hiddenDim*numExperts*dim)
	for i := 0; i < hiddenDim*dim; i++ {
		exWeightsData[i] = 1.0
	}
	for i := hiddenDim * dim; i < 2*hiddenDim*dim; i++ {
		exWeightsData[i] = 2.0
	}
	_ = expertWeights.LoadFrom(exWeightsData)
	defer expertWeights.Free()

	indices := ctx.NewTensorFP32(batchSize, topK)
	_ = indices.LoadFrom([]float32{0, 1})
	defer indices.Free()

	weights := ctx.NewTensorFP32(batchSize, topK)
	_ = weights.LoadFrom([]float32{0.4, 0.6})
	defer weights.Free()

	output := ctx.MOEExpertForward(input, expertWeights, indices, weights, hiddenDim)
	defer output.Free()

	ctx.Synchronize()
	outData := output.ToHost()
	t.Logf("Expert Forward Output: %v", outData)

	// Expert 0 dot product = 4 * 1.0 = 4.0
	// Expert 1 dot product = 4 * 2.0 = 8.0
	// Combined = 0.4 * 4.0 + 0.6 * 8.0 = 1.6 + 4.8 = 6.4
	for i := 0; i < hiddenDim; i++ {
		if math.Abs(float64(outData[i]-6.4)) > 1e-3 {
			t.Errorf("Output[%d]: expected 6.4, got %f", i, outData[i])
		}
	}
}

func TestCUDA_MOEExpertGateUpSwiGLU(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// 2 tokens, 2 experts, dim=2, hidden_dim=2, top_k=1
	batchSize := 2
	numExperts := 2
	dim := 2
	hiddenDim := 2
	topK := 1

	input := ctx.NewTensorFP32(batchSize, dim)
	_ = input.LoadFrom([]float32{
		1.0, 2.0, // Token 0
		-1.0, 1.0, // Token 1
	})
	defer input.Free()

	// Gate weights: [num_experts * hidden_dim, dim]
	// Expert 0: [[1, 0], [0, 1]]
	// Expert 1: [[0.5, 0.5], [1, -1]]
	gateWeights := ctx.NewTensorFP32(numExperts*hiddenDim, dim)
	_ = gateWeights.LoadFrom([]float32{
		1.0, 0.0,
		0.0, 1.0,
		0.5, 0.5,
		1.0, -1.0,
	})
	defer gateWeights.Free()

	// Up weights: [num_experts * hidden_dim, dim]
	// Expert 0: [[2, 1], [1, 2]]
	// Expert 1: [[1, 1], [2, 0]]
	upWeights := ctx.NewTensorFP32(numExperts*hiddenDim, dim)
	_ = upWeights.LoadFrom([]float32{
		2.0, 1.0,
		1.0, 2.0,
		1.0, 1.0,
		2.0, 0.0,
	})
	defer upWeights.Free()

	indices := ctx.NewTensorFP32(batchSize, topK)
	_ = indices.LoadFrom([]float32{
		0.0, // Token 0 -> Expert 0
		1.0, // Token 1 -> Expert 1
	})
	defer indices.Free()

	weights := ctx.NewTensorFP32(batchSize, topK)
	_ = weights.LoadFrom([]float32{
		1.0,
		1.0,
	})
	defer weights.Free()

	output := ctx.MOEExpertGateUpSwiGLU(input, gateWeights, upWeights, indices, weights, hiddenDim)
	defer output.Free()

	ctx.Synchronize()
	outData := output.ToHost()
	t.Logf("GateUpSwiGLU Output: %v", outData)

	// Token 0 -> Expert 0:
	// in = [1, 2]
	// h=0: gate = 1*1 + 2*0 = 1.0; up = 1*2 + 2*1 = 4.0; silu(1.0) = 1.0 / (1 + exp(-1.0)) ≈ 0.7310586; out = 4.0 * 0.7310586 ≈ 2.92423
	// h=1: gate = 1*0 + 2*1 = 2.0; up = 1*1 + 2*2 = 5.0; silu(2.0) = 2.0 / (1 + exp(-2.0)) ≈ 1.761594; out = 5.0 * 1.761594 ≈ 8.80797
	silu := func(x float32) float32 {
		return x / (1.0 + float32(math.Exp(float64(-x))))
	}
	expTok0H0 := 4.0 * silu(1.0)
	expTok0H1 := 5.0 * silu(2.0)

	if math.Abs(float64(outData[0]-expTok0H0)) > 1e-3 {
		t.Errorf("Token 0 H0: expected %f, got %f", expTok0H0, outData[0])
	}
	if math.Abs(float64(outData[1]-expTok0H1)) > 1e-3 {
		t.Errorf("Token 0 H1: expected %f, got %f", expTok0H1, outData[1])
	}
}
