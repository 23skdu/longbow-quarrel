//go:build (!cuda && !metal && !tpu) || !amd64 || !cgo || (!linux && !darwin)

package device

import (
	"math"
	"testing"
)

func TestCPU_MOERouterLogits(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	batchSize := 2
	numExperts := 4
	dim := 4

	input := ctx.NewTensorFP32(batchSize, dim)
	_ = input.LoadFrom([]float32{
		1.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0,
	})

	gateWeight := ctx.NewTensorFP32(numExperts, dim)
	_ = gateWeight.LoadFrom([]float32{
		1.0, 0.0, 0.0, 0.0,
		0.5, 0.5, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 1.0,
	})

	logits := ctx.MOERouterLogits(input, gateWeight)
	logitData := logits.ToHost()

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

func TestCPU_MOETopKSelection(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	batchSize := 2
	numExperts := 4
	topK := 2

	logits := ctx.NewTensorFP32(batchSize, numExperts)
	_ = logits.LoadFrom([]float32{
		1.0, 0.5, 0.0, 0.0,
		0.0, 0.5, 1.0, 0.0,
	})

	indices, weights := ctx.MOETopKSelection(logits, topK)
	idxData := indices.ToHost()
	wData := weights.ToHost()

	if int32(idxData[0]) != 0 || int32(idxData[1]) != 1 {
		t.Errorf("Token 0: expected experts [0, 1], got [%v, %v]", idxData[0], idxData[1])
	}
	if int32(idxData[2]) != 2 || int32(idxData[3]) != 1 {
		t.Errorf("Token 1: expected experts [2, 1], got [%v, %v]", idxData[2], idxData[3])
	}

	exp0 := float32(math.Exp(1.0) / (math.Exp(1.0) + math.Exp(0.5)))
	exp1 := float32(math.Exp(0.5) / (math.Exp(1.0) + math.Exp(0.5)))
	if math.Abs(float64(wData[0]-exp0)) > 1e-4 {
		t.Errorf("Token 0 w[0]: expected %f, got %f", exp0, wData[0])
	}
	if math.Abs(float64(wData[1]-exp1)) > 1e-4 {
		t.Errorf("Token 0 w[1]: expected %f, got %f", exp1, wData[1])
	}
}

func TestCPU_MOEExpertForward(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	batchSize := 1
	numExperts := 2
	dim := 4
	hiddenDim := 4
	topK := 2

	input := ctx.NewTensorFP32(batchSize, dim)
	_ = input.LoadFrom([]float32{1, 1, 1, 1})

	expertWeights := ctx.NewTensorFP32(hiddenDim*numExperts, dim)
	exWeightsData := make([]float32, hiddenDim*numExperts*dim)
	for i := 0; i < hiddenDim*dim; i++ {
		exWeightsData[i] = 1.0
	}
	for i := hiddenDim * dim; i < 2*hiddenDim*dim; i++ {
		exWeightsData[i] = 2.0
	}
	_ = expertWeights.LoadFrom(exWeightsData)

	indices := ctx.NewTensorFP32(batchSize, topK)
	_ = indices.LoadFrom([]float32{0, 1})

	weights := ctx.NewTensorFP32(batchSize, topK)
	_ = weights.LoadFrom([]float32{0.4, 0.6})

	output := ctx.MOEExpertForward(input, expertWeights, indices, weights, hiddenDim)
	outData := output.ToHost()

	for i := 0; i < hiddenDim; i++ {
		if math.Abs(float64(outData[i]-6.4)) > 1e-3 {
			t.Errorf("Output[%d]: expected 6.4, got %f", i, outData[i])
		}
	}
}

func TestCPU_MOEExpertGateUpSwiGLU(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	batchSize := 2
	numExperts := 2
	dim := 2
	hiddenDim := 2
	topK := 1

	input := ctx.NewTensorFP32(batchSize, dim)
	_ = input.LoadFrom([]float32{
		1.0, 2.0,
		-1.0, 1.0,
	})

	gateWeights := ctx.NewTensorFP32(numExperts*hiddenDim, dim)
	_ = gateWeights.LoadFrom([]float32{
		1.0, 0.0,
		0.0, 1.0,
		0.5, 0.5,
		1.0, -1.0,
	})

	upWeights := ctx.NewTensorFP32(numExperts*hiddenDim, dim)
	_ = upWeights.LoadFrom([]float32{
		2.0, 1.0,
		1.0, 2.0,
		1.0, 1.0,
		2.0, 0.0,
	})

	indices := ctx.NewTensorFP32(batchSize, topK)
	_ = indices.LoadFrom([]float32{
		0.0,
		1.0,
	})

	weights := ctx.NewTensorFP32(batchSize, topK)
	_ = weights.LoadFrom([]float32{
		1.0,
		1.0,
	})

	output := ctx.MOEExpertGateUpSwiGLU(input, gateWeights, upWeights, indices, weights, hiddenDim)
	outData := output.ToHost()

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
