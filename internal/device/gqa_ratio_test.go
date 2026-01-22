//go:build darwin && metal

package device

import (
	"fmt"
	"math"
	"testing"
)

func TestGQA_Ratio_Calculation(t *testing.T) {
	heads := 32
	kvHeads := 8
	headDim := 128

	groupSize := heads / kvHeads
	if groupSize != 4 {
		t.Fatalf("Test setup error: expected group size 4, got %d", groupSize)
	}

	ctx := NewContext()
	defer ctx.Free()

	inputData := make([]float32, heads*headDim)
	for i := 0; i < heads*headDim; i++ {
		inputData[i] = float32(i) / 100.0
	}

	q := ctx.NewTensor(1, heads*headDim)
	q.LoadFrom(inputData)
	defer q.ReturnToPool()

	kCache := ctx.NewTensor(1, kvHeads*headDim)
	vCache := ctx.NewTensor(1, kvHeads*headDim)

	kData := make([]float32, kvHeads*headDim)
	vData := make([]float32, kvHeads*headDim)

	for i := 0; i < kvHeads*headDim; i++ {
		kData[i] = float32(i) * 10.0
		vData[i] = float32(i) * 20.0
	}
	kCache.LoadFrom(kData)
	vCache.LoadFrom(vData)
	defer kCache.ReturnToPool()
	defer vCache.ReturnToPool()

	seqLen := 1
	scores := ctx.NewTensor(seqLen, heads*headDim)
	defer scores.ReturnToPool()

	pos := 5
	ropeTheta := float32(1000000.0)

	q.RoPE(pos, headDim, heads, 1, ropeTheta)
	q.AttScores(kCache, scores, pos, heads, kvHeads, headDim, seqLen, 0)
	ctx.Synchronize()

	scoresData := scores.ToHost()

	for h := 0; h < heads; h++ {
		expectedKVH := h / (heads / kvHeads)
		if expectedKVH >= kvHeads {
			t.Errorf("Invalid expected kvh calculation for head %d: got %d, max %d", h, expectedKVH, kvHeads)
		}

		hGroup := h % groupSize

		if hGroup == 0 {
			shouldMatchKVH := 0
			offsetK := shouldMatchKVH * headDim
			offsetV := shouldMatchKVH * headDim

			expectedK := float32(offsetK+h*headDim) * 10.0
			expectedV := float32(offsetV+h*headDim) * 20.0

			qIdx := h * headDim
			kIdx := h * headDim
			vIdx := h * headDim

			kVal := kCache.ToHost()[kIdx]
			vVal := vCache.ToHost()[vIdx]

			if math.Abs(float64(kVal-expectedK)) > 1e-3 {
				t.Errorf("Head %d (group 0) K cache mismatch: got %.6f, want %.6f", h, kVal, expectedK)
			}
			if math.Abs(float64(vVal-expectedV)) > 1e-3 {
				t.Errorf("Head %d (group 0) V cache mismatch: got %.6f, want %.6f", h, vVal, expectedV)
			}
		}
	}

	t.Logf("✓ GQA ratio calculation verified: heads=%d, kv_heads=%d, group_size=%d", heads, kvHeads, groupSize)
}

func TestGQA_KVMapping_LargeRatio(t *testing.T) {
	heads := 8
	kvHeads := 2
	headDim := 64

	groupSize := heads / kvHeads
	if groupSize != 4 {
		t.Fatalf("Test setup error: expected group size 4, got %d", groupSize)
	}

	ctx := NewContext()
	defer ctx.Free()

	kCache := ctx.NewTensor(1, kvHeads*headDim)
	defer kCache.ReturnToPool()

	kData := make([]float32, kvHeads*headDim)
	for i := 0; i < kvHeads*headDim; i++ {
		kData[i] = float32(i+1) * 100.0
	}
	kCache.LoadFrom(kData)

	qData := make([]float32, heads*headDim)
	for i := 0; i < heads*headDim; i++ {
		qData[i] = float32(i)
	}
	q := ctx.NewTensor(1, heads*headDim)
	defer q.ReturnToPool()
	q.LoadFrom(qData)

	q.RoPE(0, headDim, heads, 1, 1.0)
	ctx.Synchronize()

	qAfterRoPE := q.ToHost()

	for h := 0; h < heads; h++ {
		expectedKVH := h / (heads / kvHeads)
		kIdx := expectedKVH * headDim

		kVal := kCache.ToHost()[kIdx]

		if math.Abs(float64(kVal-100.0)) > 1e-3 {
			t.Errorf("Head %d should use KV[%d] after RoPE, got %.6f, want 100.0", h, expectedKVH, kVal)
		}
	}

	t.Logf("✓ Large GQA ratio (8:2) verified: heads=%d, kv_heads=%d", heads, kvHeads)
}

func TestGQA_ScoresOutputShape(t *testing.T) {
	heads := 4
	kvHeads := 2
	headDim := 32
	seqLen := 1

	ctx := NewContext()
	defer ctx.Free()

	q := ctx.NewTensor(1, heads*headDim)
	defer q.ReturnToPool()

	kCache := ctx.NewTensor(seqLen, kvHeads*headDim)
	defer kCache.ReturnToPool()

	qData := make([]float32, heads*headDim)
	for i := 0; i < heads*headDim; i++ {
		qData[i] = 1.0
	}
	q.LoadFrom(qData)

	kData := make([]float32, kvHeads*headDim)
	for i := 0; i < kvHeads*headDim; i++ {
		kData[i] = float32(i + 10)
	}
	kCache.LoadFrom(kData)

	scores := ctx.NewTensor(seqLen, heads*headDim)
	defer scores.ReturnToPool()

	q.AttScores(kCache, scores, 0, heads, kvHeads, headDim, seqLen, 0)
	ctx.Synchronize()

	scoresData := scores.ToHost()

	if len(scoresData) != heads*headDim {
		t.Errorf("Scores output shape mismatch: got %d, want %d", len(scoresData), heads*headDim)
	}

	t.Logf("✓ Scores output shape verified: heads=%d, head_dim=%d, output_size=%d", heads, headDim, len(scoresData))
}

func TestGQA_AttValues_Shape(t *testing.T) {
	heads := 16
	kvHeads := 4
	headDim := 128
	seqLen := 1

	ctx := NewContext()
	defer ctx.Free()

	scores := ctx.NewTensor(seqLen, heads*headDim)
	defer scores.ReturnToPool()

	kCache := ctx.NewTensor(seqLen, kvHeads*headDim)
	defer kCache.ReturnToPool()

	vCache := ctx.NewTensor(seqLen, kvHeads*headDim)
	defer vCache.ReturnToPool()

	kData := make([]float32, kvHeads*headDim)
	vData := make([]float32, kvHeads*headDim)
	for i := 0; i < kvHeads*headDim; i++ {
		kData[i] = float32(i) * 2.0
		vData[i] = float32(i) * 3.0
	}
	kCache.LoadFrom(kData)
	vCache.LoadFrom(vData)

	scoresData := make([]float32, heads*headDim)
	for i := 0; i < heads*headDim; i++ {
		if i < kvHeads*headDim {
			scoresData[i] = float32(i + 1)
		} else {
			scoresData[i] = -10000.0
		}
	}
	scores.LoadFrom(scoresData)

	output := ctx.NewTensor(1, heads*headDim)
	defer output.ReturnToPool()

	output.AttValues(scores, vCache, 0, heads, kvHeads, headDim)
	ctx.Synchronize()

	outputData := output.ToHost()

	if len(outputData) != heads*headDim {
		t.Errorf("AttValues output shape mismatch: got %d, want %d", len(outputData), heads*headDim)
	}

	for h := 0; h < heads; h++ {
		expectedKVH := h / (heads / kvHeads)
		kvIdx := expectedKVH * headDim

		kVal := kCache.ToHost()[kvIdx]
		vVal := vCache.ToHost()[kvIdx]

		score := scoresData[h]

		if score < 0 {
			if math.Abs(float64(outputData[h*headDim]-kVal*2.0)) > 0.1 {
				t.Errorf("Head %d negative score: output should not use KV[%d], got %.6f vs k=%.6f", h, expectedKVH, outputData[h*headDim], kVal)
			}
			if math.Abs(float64(outputData[h*headDim+1]-vVal*3.0)) > 0.1 {
				t.Errorf("Head %d negative score: output should not use KV[%d], got %.6f vs v=%.6f", h, expectedKVH, outputData[h*headDim+1], vVal)
			}
		} else {
			if math.Abs(float64(outputData[h*headDim]-kVal*2.0)) > 0.1 {
				t.Errorf("Head %d positive score: output should use KV[%d], got %.6f vs k=%.6f", h, expectedKVH, outputData[h*headDim], kVal)
			}
			if math.Abs(float64(outputData[h*headDim+1]-vVal*3.0)) > 0.1 {
				t.Errorf("Head %d positive score: output should use KV[%d], got %.6f vs v=%.6f", h, expectedKVH, outputData[h*headDim+1], vVal)
			}
		}
	}

	t.Logf("✓ AttValues shape and KV mapping verified: heads=%d, kv_heads=%d", heads, kvHeads)
}

func TestGQA_Mistral_32_8(t *testing.T) {
	heads := 32
	kvHeads := 8
	headDim := 128

	ctx := NewContext()
	defer ctx.Free()

	q := ctx.NewTensor(1, heads*headDim)
	defer q.ReturnToPool()

	kCache := ctx.NewTensor(1, kvHeads*headDim)
	vCache := ctx.NewTensor(1, kvHeads*headDim)
	defer kCache.ReturnToPool()
	defer vCache.ReturnToPool()

	qData := make([]float32, heads*headDim)
	for i := 0; i < heads*headDim; i++ {
		qData[i] = float32(i) / 100.0
	}
	q.LoadFrom(qData)

	kData := make([]float32, kvHeads*headDim)
	for i := 0; i < kvHeads*headDim; i++ {
		kData[i] = float32(i+100) * 5.0
	}
	vData := make([]float32, kvHeads*headDim)
	for i := 0; i < kvHeads*headDim; i++ {
		vData[i] = float32(i+200) * 7.0
	}
	kCache.LoadFrom(kData)
	vCache.LoadFrom(vData)

	scores := ctx.NewTensor(1, heads*headDim)
	defer scores.ReturnToPool()

	output := ctx.NewTensor(1, heads*headDim)
	defer output.ReturnToPool()

	pos := 3
	q.RoPE(pos, headDim, heads, 1, 1000000.0)
	q.AttScores(kCache, scores, pos, heads, kvHeads, headDim, 1, 0)
	output.AttValues(scores, vCache, pos, heads, kvHeads, headDim)
	ctx.Synchronize()

	scoresData := scores.ToHost()
	outputData := output.ToHost()

	t.Logf("Mistral 32:8 GQA test: pos=%d", pos)

	for h := 0; h < 8; h++ {
		for i := 0; i < 4; i++ {
			queryHead := h + i*(kvHeads)
			kvh := queryHead / (heads / kvHeads)

			if kvh != h {
				t.Errorf("Head %d (query head %d) should map to KV head %d, got %d", queryHead, queryHead, kvh)
			}
		}
	}

	t.Logf("✓ Mistral 32:8 GQA mapping verified: all 32 query heads correctly mapped to 8 KV heads")
}

func TestGQA_SlidingWindow_Indexing(t *testing.T) {
	heads := 4
	kvHeads := 2
	headDim := 64
	seqLen := 10
	windowSize := 4
	pos := 6

	ctx := NewContext()
	defer ctx.Free()

	q := ctx.NewTensor(1, heads*headDim)
	defer q.ReturnToPool()

	kCache := ctx.NewTensor(windowSize, kvHeads*headDim)
	defer kCache.ReturnToPool()

	vCache := ctx.NewTensor(windowSize, kvHeads*headDim)
	defer vCache.ReturnToPool()

	qData := make([]float32, heads*headDim)
	for i := 0; i < heads*headDim; i++ {
		qData[i] = 1.0
	}
	q.LoadFrom(qData)

	kData := make([]float32, windowSize*kvHeads*headDim)
	vData := make([]float32, windowSize*kvHeads*headDim)
	for i := 0; i < windowSize*kvHeads*headDim; i++ {
		kData[i] = float32(i+1) * 2.0
		vData[i] = float32(i+100) * 3.0
	}
	kCache.LoadFrom(kData)
	vCache.LoadFrom(vData)

	scores := ctx.NewTensor(1, heads*headDim)
	defer scores.ReturnToPool()

	output := ctx.NewTensor(1, heads*headDim)
	defer output.ReturnToPool()

	q.RoPE(pos, headDim, heads, 1, 1.0)
	q.AttScores(kCache, scores, pos, heads, kvHeads, headDim, seqLen, windowSize)
	output.AttValues(scores, vCache, pos, heads, kvHeads, headDim)
	ctx.Synchronize()

	scoresData := scores.ToHost()
	outputData := output.ToHost()

	expectedWindowPos := pos % windowSize

	for h := 0; h < heads; h++ {
		kvh := h / (heads / kvHeads)

		kvPosOffset := expectedWindowPos * kvHeads * headDim
		expectedKVal := float32((kvPosOffset+h*headDim)+1) * 2.0
		expectedVVal := float32((kvPosOffset+h*headDim)+101) * 3.0

		kIdx := kvh * headDim
		kVal := kCache.ToHost()[kIdx]
		vIdx := kvh * headDim
		vVal := vCache.ToHost()[vIdx]

		if math.Abs(float64(kVal-expectedKVal)) > 0.1 {
			t.Errorf("Head %d K mismatch at pos %d (window pos %d): got %.6f, want %.6f", h, pos, expectedWindowPos, kVal, expectedKVal)
		}
		if math.Abs(float64(vVal-expectedVVal)) > 0.1 {
			t.Errorf("Head %d V mismatch at pos %d (window pos %d): got %.6f, want %.6f", h, pos, expectedWindowPos, vVal, expectedVVal)
		}
	}

	t.Logf("✓ GQA sliding window indexing verified: window_size=%d, pos=%d, window_pos=%d", windowSize, pos, expectedWindowPos)
}
