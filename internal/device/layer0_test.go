package device

import (
	"math"
	"math/rand"
	"testing"
)

// Helper to compare slices
func assertClose(t *testing.T, name string, cpu, gpu []float32, tolerance float64) {
	if len(cpu) != len(gpu) {
		t.Fatalf("%s: Length mismatch CPU %d vs GPU %d", name, len(cpu), len(gpu))
	}
	for i := range cpu {
		diff := math.Abs(float64(cpu[i] - gpu[i]))
		if diff > tolerance {
			t.Errorf("%s: Mismatch at index %d. CPU %f, GPU %f, Diff %f", name, i, cpu[i], gpu[i], diff)
			return // Fail fast to avoid spam
		}
	}
}

func TestLayer0_RMSNorm(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	rows := 1 // Batch = 1
	dim := 4096 // Mistral Dim
	eps := float32(1e-5)

	// 1. Generate Input and Weight
	input := make([]float32, rows*dim)
	weight := make([]float32, dim)
	
	for i := range input {
		input[i] = rand.Float32() // 0-1
	}
	for i := range weight {
		weight[i] = rand.Float32()
	}

	// 2. Run CPU Reference
	cpuOut := CPURMSNorm(input, weight, eps)

	// 3. Run GPU Implementation
	tInput := ctx.NewTensor(rows, dim)
	tInput.LoadFrom(input) // F16
	
	tWeight := ctx.NewTensor(1, dim) // 1D weight [1, Dim] or [Dim]? RMSNorm expects [Dim] usually
	tWeight.LoadFrom(weight)
	
	tOut := tInput.RMSNorm(tWeight, eps)
	
	tOut.ctx.Synchronize()
	gpuOut := tOut.ToHost()

	// RMSNorm precision can be sensitive. Tolerance 1e-2 for F16 accumulation?
	assertClose(t, "RMSNorm", cpuOut, gpuOut, 2e-3)
}

func TestLayer0_LinearF16(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Dimensions for Mistral Q/K/V projection
	M := 1 // Batch
	K := 4096 // In features (Dim)
	N := 4096 // Out features (Dim) - Simplified, usually 4096 for Q, 1024 for K/V etc.
	
	// 1. Generate Input and Weight
	input := make([]float32, M*K)
	weight := make([]float32, N*K) // [Out, In]
	
	for i := range input {
		input[i] = (rand.Float32() - 0.5) * 0.1
	}
	for i := range weight {
		weight[i] = (rand.Float32() - 0.5) * 0.1
	}

	// 2. Run CPU Reference (A * B^T)
	cpuOut := CPUMatMul(input, weight, M, N, K)

	// 3. Run GPU Implementation
	// Linear expects weight tensor [N, K]
	// Input tensor [M, K]
	// Output [M, N]
	
	tInput := ctx.NewTensor(M, K)
	tInput.LoadFrom(input)
	
	tWeight := ctx.NewTensor(N, K)
	tWeight.LoadFrom(weight)
	
	// We use Linear, which returns new tensor
	// Linear calls BatchedMatMul_F16 if weight is F16
	tOut := tInput.Linear(tWeight)
	
	tOut.ctx.Synchronize()
	gpuOut := tOut.ToHost() // [M, N]
	// 4. Compare
	assertClose(t, "LinearF16", cpuOut, gpuOut, 1e-2)
}

func TestLayer0_RoPE(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Dimensions
	Batch := 1
	Heads := 32
	HeadDim := 128
	Dim := Heads * HeadDim
	Theta := float32(10000.0) // Mistral used 10000.0 (or 1M? standard is 10k usually)
	
	// 1. Input
	input := make([]float32, Batch*Dim)
	for i := range input {
		input[i] = rand.Float32() * 0.1
	}
	
	// 2. CPU
	cpuOut := CPURoPE(input, 0, Heads, HeadDim, Theta)
	
	// 3. GPU
	tInput := ctx.NewTensor(Batch, Dim)
	tInput.LoadFrom(input)
	
	// Call RoPE (Method modifies in place)
	// Arguments: pos, headDim, numHeads, seqLen, theta
	tInput.RoPE(0, HeadDim, Heads, 1, Theta)
	
	tInput.ctx.Synchronize()
	gpuOut := tInput.ToHost()
	// 4. Compare
	assertClose(t, "RoPE", cpuOut, gpuOut, 1e-3)
}

func TestLayer0_Attention(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Dimensions for single token attention
	Heads := 4
	KVHeads := 1 // GQA
	HeadDim := 32
	Pos := 5 // 5th token
	CtxLen := 10 // Padded context
	Scale := float32(1.0 / math.Sqrt(float64(HeadDim)))
	
	// 1. Inputs
	// Q: [Heads * HeadDim] (Current token)
	qData := make([]float32, Heads*HeadDim)
	for i := range qData { qData[i] = rand.Float32() * 0.5 }
	
	// K Cache: [CtxLen, KVHeads*HeadDim] (Rows are tokens)
	kData := make([]float32, CtxLen*KVHeads*HeadDim)
	for i := range kData { kData[i] = rand.Float32() * 0.5 }
	
	// V Cache: Same as K
	vData := make([]float32, CtxLen*KVHeads*HeadDim)
	for i := range vData { vData[i] = rand.Float32() * 0.5 }
	
	// 2. CPU Reference
	// Loop over heads
	cpuOut := make([]float32, Heads*HeadDim)
	
	for h := 0; h < Heads; h++ {
		kvH := h / (Heads / KVHeads)
		// For each previous token t <= Pos
		scores := make([]float32, Pos+1)
		
		// 2a. Score = Q . K * Scale
		for t := 0; t <= Pos; t++ {
			var dot float32 = 0
			for d := 0; d < HeadDim; d++ {
				qVal := qData[h*HeadDim+d]
				kVal := kData[t*(KVHeads*HeadDim) + kvH*HeadDim + d]
				dot += qVal * kVal
			}
			scores[t] = dot * Scale
		}
		
		// 2b. Softmax
		// Find max
		maxVal := float32(-math.MaxFloat32)
		for t := 0; t <= Pos; t++ {
			if scores[t] > maxVal { maxVal = scores[t] }
		}
		var sumExp float32 = 0
		for t := 0; t <= Pos; t++ {
			scores[t] = float32(math.Exp(float64(scores[t] - maxVal)))
			sumExp += scores[t]
		}
		for t := 0; t <= Pos; t++ {
			scores[t] /= sumExp
		}
		
		// 2c. Output = Score . V
		for d := 0; d < HeadDim; d++ {
			var val float32 = 0
			for t := 0; t <= Pos; t++ {
				vVal := vData[t*(KVHeads*HeadDim) + kvH*HeadDim + d]
				val += scores[t] * vVal
			}
			cpuOut[h*HeadDim+d] = val
		}
	}
	
	// 3. GPU
	tQ := ctx.NewTensor(1, Heads*HeadDim)
	tQ.LoadFrom(qData)
	
	tKCache := ctx.NewTensor(CtxLen, KVHeads*HeadDim) // Rows=CtxLen, Cols=Dim
	tKCache.LoadFrom(kData)
	
	tVCache := ctx.NewTensor(CtxLen, KVHeads*HeadDim)
	tVCache.LoadFrom(vData)
	
	// Attention(k, v, pos, numHeads, kvHeads, headDim, ctxLen)
	tOut := tQ.Attention(tKCache, tVCache, Pos, Heads, KVHeads, HeadDim, CtxLen)
	
	tOut.ctx.Synchronize()
	gpuOut := tOut.ToHost()
	
	// 4. Compare
	// Softmax accumulation in half precision is noisy.
	assertClose(t, "Attention", cpuOut, gpuOut, 2e-2)
}

func TestLayer0_SwiGLU(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Dimensions
	M := 1
	Dim := 128
	
	// 1. Input
	// SwiGLU takes two inputs: Gate and Up (Val)
	gate := make([]float32, M*Dim)
	up := make([]float32, M*Dim)
	
	for i := range gate {
		gate[i] = (rand.Float32() - 0.5) * 2.0 // Range -1 to 1 for Silu check
		up[i] = (rand.Float32() - 0.5)
	}
	
	// 2. CPU
	cpuOut := CPUSwiGLU(gate, up)
	
	// 3. GPU
	tUp := ctx.NewTensor(M, Dim)
	tUp.LoadFrom(up)
	
	tGate := ctx.NewTensor(M, Dim)
	tGate.LoadFrom(gate)
	
	// SwiGLU(gate) -> returns new tensor with result
	// The receiver (t) is 'up', arg is 'gate'
	tOut := tUp.SwiGLU(tGate)
	
	tOut.ctx.Synchronize()
	gpuOut := tOut.ToHost()
	
	// 4. Compare
	assertClose(t, "SwiGLU", cpuOut, gpuOut, 1e-3)
}
