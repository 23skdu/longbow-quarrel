//go:build darwin && metal
package device

import (
	"math"
	"testing"
)

func TestFlashAttention2_NumericalParity(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// 1. Hyperparameters
	numHeads := 4
	kvHeads := 4
	headDim := 32
	seqLen := 32
	blockSize := 16
	batchSize := 1
	maxBlocks := (seqLen + blockSize - 1) / blockSize

	// 2. Tensors
	q := ctx.NewTensor(numHeads, headDim)
	kCache := ctx.NewTensor(maxBlocks * blockSize * kvHeads, headDim)
	vCache := ctx.NewTensor(maxBlocks * blockSize * kvHeads, headDim)
	output := ctx.NewTensor(numHeads, headDim)
	blockTable := ctx.NewTensor(batchSize, maxBlocks)

	// 3. Fill Test Data (smaller numbers to avoid exp overflow)
	qData := make([]float32, numHeads*headDim)
	for i := range qData { qData[i] = float32(i % headDim) * 0.001 }
	q.LoadFrom(qData)

	kData := make([]float32, kCache.Rows()*headDim)
	for i := range kData { kData[i] = float32(i % headDim) * 0.0005 }
	kCache.LoadFrom(kData)

	vData := make([]float32, vCache.Rows()*headDim)
	for i := range vData { vData[i] = float32(i % headDim) * 0.0002 }
	vCache.LoadFrom(vData)

	// Logical map: block i -> physical block i
	btData := make([]float32, maxBlocks)
	for i := range btData { btData[i] = float32(i) }
	blockTable.LoadFrom(btData)

	seqLens := ctx.NewTensorFP32(1, batchSize)
	_ = seqLens.LoadFrom([]float32{float32(seqLen)})
	tokenToSeq := ctx.NewTensorFP32(1, 1) // 1 token (decode case)
	_ = tokenToSeq.LoadFrom([]float32{0})

	// 4. Run FlashAttention2
	ctx.FlashAttention2(q, kCache, vCache, output, seqLens, numHeads, kvHeads, headDim, blockSize, blockTable, maxBlocks, tokenToSeq, batchSize)
	ctx.Synchronize()

	flashResult := output.ToHost()

	// 5. Reference Implementation (CPU-style simplified)
	refOutput := make([]float32, numHeads*headDim)
	scale := 1.0 / math.Sqrt(float64(headDim))

	for h := 0; h < numHeads; h++ {
		m_i := float64(-1e20)
		l_i := float64(0.0)
		acc_o := make([]float64, headDim)

		for s := 0; s < seqLen; s++ {
			// Physical mapping for token s
			blockIdx := s / blockSize
			tokenIdx := s % blockSize
			kvOff := blockIdx*blockSize*kvHeads*headDim +
				tokenIdx*kvHeads*headDim +
				(h / (numHeads / kvHeads))*headDim

			// Compute score S_ij = Q @ K
			dot := float64(0)
			for d := 0; d < headDim; d++ {
				qVal := float64(qData[h*headDim+d])
				kVal := float64(kData[kvOff+d])
				dot += qVal * kVal
			}
			score := dot * scale

			// Online Softmax
			m_prev := m_i
			m_i = math.Max(m_prev, score)
			alpha := math.Exp(m_prev - m_i)
			exp_s := math.Exp(score - m_i)
			l_i = l_i*alpha + exp_s

			// Update Accumulator
			vOff := kvOff // V layout matches K
			for d := 0; d < headDim; d++ {
				vVal := float64(vData[vOff+d])
				acc_o[d] = acc_o[d]*alpha + exp_s*vVal
			}
		}

		// Finalize
		for d := 0; d < headDim; d++ {
			refOutput[h*headDim+d] = float32(acc_o[d] / l_i)
		}
	}

	// 6. Compare
	passed := true
	for i := range flashResult {
		diff := math.Abs(float64(flashResult[i] - refOutput[i]))
		if diff > 1e-3 {
			if passed {
				t.Logf("First mismatch at index %d: flash=%f, ref=%f, diff=%e", i, flashResult[i], refOutput[i], diff)
				passed = false
			}
		}
	}
	if !passed {
		t.Fail()
	}
}
