//go:build darwin && metal

package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestMOERouting(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	// Test case: 2 tokens, 4 experts, dim=4, top_k=2
	batchSize := 2
	numExperts := 4
	dim := 4
	topK := 2

	// Input and Gate weights should be F16 (NewTensor)
	input := ctx.NewTensor(batchSize, dim)
	input.LoadFrom([]float32{
		1.0, 0.0, 0.0, 0.0, // Token 0
		0.0, 1.0, 0.0, 0.0, // Token 1
	})
	defer input.Free()

	gateWeight := ctx.NewTensor(numExperts, dim)
	gateWeight.LoadFrom([]float32{
		1.0, 0.0, 0.0, 0.0, // Expert 0 (perfect match for token 0)
		0.5, 0.5, 0.0, 0.0, // Expert 1
		0.0, 1.0, 0.0, 0.0, // Expert 2 (perfect match for token 1)
		0.0, 0.0, 1.0, 1.0, // Expert 3
	})
	defer gateWeight.Free()

	// 1. Compute logits (Output is F32)
	logits := ctx.MOERouterLogits(input, gateWeight)
	defer logits.Free()

	logitData := logits.ToHost()
	t.Logf("Logits: %v", logitData)

	// Token 0 logits: [1.0, 0.5, 0.0, 0.0]
	// Token 1 logits: [0.0, 0.5, 1.0, 0.0]

	// 2. Select top-k (Inputs and Outputs are F32)
	expertIndices, expertWeights := ctx.MOETopKSelection(logits, topK)
	defer expertIndices.Free()
	defer expertWeights.Free()

	indicesData := expertIndices.ToHost()
	weightsData := expertWeights.ToHost()
	t.Logf("Expert Indices: %v", indicesData)
	t.Logf("Expert Weights: %v", weightsData)

	// Token 0: top experts should be 0 and 1
	if int32(indicesData[0]) != 0 || int32(indicesData[1]) != 1 {
		t.Errorf("Token 0: expected experts [0, 1], got [%v, %v]", indicesData[0], indicesData[1])
	}

	// Token 1: top experts should be 2 and 1
	if int32(indicesData[2]) != 2 || int32(indicesData[3]) != 1 {
		t.Errorf("Token 1: expected experts [2, 1], got [%v, %v]", indicesData[2], indicesData[3])
	}

	// Verify softmax weights
	// For token 0: exp(1.0) / (exp(1.0) + exp(0.5)) vs exp(0.5) / (exp(1.0) + exp(0.5))
	// weight[0] > weight[1]
	if weightsData[0] <= weightsData[1] {
		t.Errorf("Token 0: weight[0] (%v) should be > weight[1] (%v)", weightsData[0], weightsData[1])
	}
}

func TestMOEExpertForward(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	// 1 token, 2 experts, dim=4, hidden_dim=4, top_k=2
	batchSize := 1
	numExperts := 2
	dim := 4
	hiddenDim := 4
	topK := 2

	// Input should be F16
	input := ctx.NewTensor(batchSize, dim)
	input.LoadFrom([]float32{1, 1, 1, 1})
	defer input.Free()

	// Experts weights: [hidden_dim * num_experts, dim] - F16
	expertWeights := ctx.NewTensor(hiddenDim*numExperts, dim)
	exWeightsData := make([]float32, hiddenDim*numExperts*dim)
	for i := 0; i < hiddenDim*dim; i++ {
		exWeightsData[i] = 1.0
	}
	for i := hiddenDim * dim; i < 2*hiddenDim*dim; i++ {
		exWeightsData[i] = 2.0
	}
	expertWeights.LoadFrom(exWeightsData)
	defer expertWeights.Free()

	// Indices and selection weights are F32
	indices := ctx.NewTensorFP32(batchSize, topK)
	indices.LoadFrom([]float32{0, 1})
	defer indices.Free()

	weights := ctx.NewTensorFP32(batchSize, topK)
	weights.LoadFrom([]float32{0.4, 0.6})
	defer weights.Free()

	// Final output should be F16
	output := ctx.MOEExpertForward(input, expertWeights, indices, weights, hiddenDim)
	defer output.Free()

	outData := output.ToHost()
	t.Logf("Expert Forward Output: %v", outData)

	// Expert 0 output: 1*1+1*1+1*1+1*1 = 4 (for each hidden dim)
	// Expert 1 output: 1*2+1*2+1*2+1*2 = 8 (for each hidden dim)
	// Weighted output: 0.4 * 4 + 0.6 * 8 = 1.6 + 4.8 = 6.4
	for i := 0; i < hiddenDim; i++ {
		if outData[i] < 6.3 || outData[i] > 6.5 {
			t.Errorf("Output[%d]: expected 6.4, got %v", i, outData[i])
		}
	}
}
