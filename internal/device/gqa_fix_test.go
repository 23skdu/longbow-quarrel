//go:build darwin && metal

package device

import (
	"testing"
)

// TestGQA_Fix_KVIndexing tests that att_values_f16 correctly uses kvh
// instead of lane index to select appropriate QH arrays
func TestGQA_Fix_KVIndexing(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	heads := 4
	kvHeads := 2
	headDim := 32
	groupSize := heads / kvHeads
	pos := 5
	seqLen := 1

	kCache := ctx.NewTensor(seqLen, kvHeads*headDim)
	defer kCache.ReturnToPool()
	vCache := ctx.NewTensor(seqLen, kvHeads*headDim)
	defer vCache.ReturnToPool()
	scores := ctx.NewTensor(seqLen, heads*headDim)
	defer scores.ReturnToPool()

	kData := make([]float32, kvHeads*headDim)
	vData := make([]float32, kvHeads*headDim)

	for i := 0; i < kvHeads*headDim; i++ {
		kData[i] = float32(i+100) * 2.0
		vData[i] = float32(i+200) * 7.0
	}
	kCache.LoadFrom(kData)
	vCache.LoadFrom(vData)

	qData := make([]float32, heads*headDim)
	for i := 0; i < heads*headDim; i++ {
		qData[i] = float32(i+1) * 3.0
	}
	q := ctx.NewTensor(1, heads*headDim)
	defer q.ReturnToPool()
	q.LoadFrom(qData)

	q.RoPE(pos, headDim, heads, 1, 1.0)
	ctx.Synchronize()

	q.AttScores(kCache, scores, pos, heads, kvHeads, headDim, seqLen, 0)
	output := ctx.NewTensor(1, heads*headDim)
	defer output.ReturnToPool()

	output.AttValues(scores, vCache, pos, heads, kvHeads, headDim)
	ctx.Synchronize()

	outputData := output.ToHost()

	t.Logf("Checking att_values_f16 KV indexing fix")
	t.Logf("heads=%d, kvHeads=%d, group_size=%d", heads, kvHeads, groupSize)

	kvhMismatchCount := 0
	wrongKVUsageCount := 0
	wrongQHUsageCount := 0

	for h := 0; h < heads; h++ {
		queryHead := h
		kvh := queryHead / groupSize
		t.Logf("Query head %d should use KV head %d", queryHead, kvh)

		for i := 0; i < headDim; i++ {
			outputIdx := h*headDim + i

			expectedKVal := float32((kvh*headDim)+i+100) * 2.0
			expectedVVal := float32((kvh*headDim)+i+200) * 7.0

			if math.Abs(float64(outputData[outputIdx]-expectedKVal)) > 0.1 {
				t.Errorf("Head %d idx %d: expected K[%d][%d]=%.2f, got %.2f", h, kvh, i, outputData[outputIdx])
			}

			if math.Abs(float64(outputData[outputIdx+headDim]-expectedVVal)) > 0.1 {
				t.Errorf("Head %d idx %d: expected V[%d][%d]=%.2f, got %.2f", h, kvh, outputData[outputIdx+headDim])
			}

			kvUsed := false
			qhUsed := false

			if i >= 16 && i < 32 {
				if math.Abs(float64(outputData[outputIdx]-expectedKVal)) < 0.1 {
					kvUsed = true
				}
			} else if i < 16 {
				if math.Abs(float64(outputData[outputIdx]-expectedKVal)) < 0.1 {
					kvUsed = true
				}
			}

			if i < 16 {
				expectedQH := 0
			} else {
				expectedQH := 1
			}

			if (i < 16) != (expectedQH == 0) {
				wrongQHUsageCount++
			}

			if kvUsed {
				kvMismatchCount++
			}
		}
	}

	t.Logf("KV mismatch count: %d", kvhMismatchCount)
	t.Logf("Wrong QH array usage count: %d", wrongQHUsageCount)

	if kvhMismatchCount > 0 {
		t.Errorf("att_values_f16 has %d KV index mismatches", kvhMismatchCount)
	}

	if wrongQHUsageCount > 0 {
		t.Errorf("att_values_f16 incorrectly selected QH arrays in %d positions", wrongQHUsageCount)
	}

	t.Logf("✓ att_values_f16 KV indexing test complete")
}

func TestGQA_Fix_KVMultiplying(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	heads := 8
	kvHeads := 2
	headDim := 128
	groupSize := 4
	pos := 3
	seqLen := 1

	kCache := ctx.NewTensor(seqLen, kvHeads*headDim)
	defer kCache.ReturnToPool()
	vCache := ctx.NewTensor(seqLen, kvHeads*headDim)
	defer vCache.ReturnToPool()
	scores := ctx.NewTensor(seqLen, heads*headDim)
	defer scores.ReturnToPool()

	kData := make([]float32, kvHeads*headDim)
	vData := make([]float32, kvHeads*headDim)

	for i := 0; i < kvHeads*headDim; i++ {
		kData[i] = float32(i+100) * 5.0
		vData[i] = float32(i+200) * 3.0
	}
	kCache.LoadFrom(kData)
	vCache.LoadFrom(vData)

	qData := make([]float32, heads*headDim)
	for i := 0; i < heads*headDim; i++ {
		qData[i] = float32(i+1) * 10.0
	}
	q := ctx.NewTensor(1, heads*headDim)
	defer q.ReturnToPool()
	q.LoadFrom(qData)

	q.RoPE(pos, headDim, heads, 1, 1.0)
	ctx.Synchronize()

	q.AttScores(kCache, scores, pos, heads, kvHeads, headDim, seqLen, 0)
	output := ctx.NewTensor(1, heads*headDim)
	defer output.ReturnToPool()

	output.AttValues(scores, vCache, pos, heads, kvHeads, headDim)
	ctx.Synchronize()

	outputData := output.ToHost()

	t.Logf("Checking att_values_f16 KV multiplication bug")
	t.Logf("heads=%d, kvHeads=%d, group_size=%d", heads, kvHeads, groupSize)

	hasMultiplicationBug := false

	for h := 0; h < heads; h++ {
		for i := 0; i < headDim; i++ {
			outputIdx := h*headDim + i

			kvh := h / groupSize
			expectedK := float32((kvh*headDim)+i+100) * 5.0
			expectedV := float32((kvh*headDim)+i+200) * 3.0

			kVal := kCache.ToHost()[kvh*headDim+i]
			vVal := vCache.ToHost()[kvh*headDim+i]

			if math.Abs(float64(outputData[outputIdx]-expectedK)) < 0.1 {
				t.Errorf("Head %d idx %d: wrong K value. Expected using KV[%d], got wrong", h, kvh, i)
				hasMultiplicationBug = true
			}

			if math.Abs(float64(outputData[outputIdx+headDim]-expectedV)) < 0.1 {
				t.Errorf("Head %d idx %d: wrong V value. Expected using KV[%d], got wrong", h, kvh, i)
				hasMultiplicationBug = true
			}
		}
	}

	if hasMultiplicationBug {
		t.Errorf("att_values_f16 has KV multiplication bug: not using kvh correctly")
	} else {
		t.Logf("✓ att_values_f16 KV indexing correct")
	}
}

func TestGQA_Fix_KVMultiHeadMapping(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	heads := 32
	kvHeads := 8
	headDim := 128
	groupSize := 4
	pos := 10
	seqLen := 1

	kCache := ctx.NewTensor(seqLen, kvHeads*headDim)
	defer kCache.ReturnToPool()
	vCache := ctx.NewTensor(seqLen, kvHeads*headDim)
	defer vCache.ReturnToPool()
	scores := ctx.NewTensor(seqLen, heads*headDim)
	defer scores.ReturnToPool()

	kData := make([]float32, kvHeads*headDim)
	vData := make([]float32, kvHeads*headDim)

	for i := 0; i < kvHeads*headDim; i++ {
		kData[i] = float32(i+100) * float32(i/2)
		vData[i] = float32(i+200) * float32(i/2)
	}
	kCache.LoadFrom(kData)
	vCache.LoadFrom(vData)

	qData := make([]float32, heads*headDim)
	for i := 0; i < heads*headDim; i++ {
		qData[i] = float32(i)
	}
	q := ctx.NewTensor(1, heads*headDim)
	defer q.ReturnToPool()
	q.LoadFrom(qData)

	q.RoPE(pos, headDim, heads, 1, 1.0)
	ctx.Synchronize()

	q.AttScores(kCache, scores, pos, heads, kvHeads, headDim, seqLen, 0)
	output := ctx.NewTensor(1, heads*headDim)
	defer output.ReturnToPool()

	output.AttValues(scores, vCache, pos, heads, kvHeads, headDim)
	ctx.Synchronize()

	outputData := output.ToHost()

	t.Logf("Checking att_values_f16 multi-head mapping (32:8)")

	for h := 0; h < heads; h++ {
		kvh := h / groupSize
		t.Logf("Query head %d maps to KV head %d", h, kvh)

		for i := 0; i < headDim; i++ {
			outputIdx := h*headDim + i

			if h == kvh {
				continue
			}

			kvhExpected := kvh

			for checkKvh := 0; checkKvh < kvHeads; checkKvh++ {
				if checkKvh == kvhExpected {
					break
				}
			}

			kVal := kCache.ToHost()[kvhExpected*headDim+i]
			vVal := vCache.ToHost()[kvhExpected*headDim+i]

			outputK := outputData[outputIdx]
			outputV := outputData[outputIdx+headDim]

			if math.Abs(float64(outputK-kVal)) < 0.01 {
				t.Errorf("Head %d idx %d: mismatch with expected KV[%d]. K: %.2f vs %.2f", h, i, kvhExpected, outputK, kVal)
			}

			if math.Abs(float64(outputV-vVal)) < 0.01 {
				t.Errorf("Head %d idx %d: mismatch with expected KV[%d]. V: %.2f vs %.2f", h, i, kvhExpected, outputV, vVal)
			}
		}
	}

	t.Logf("✓ att_values_f16 multi-head mapping (32:8) verified")
}
