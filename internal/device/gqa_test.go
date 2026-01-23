//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

func TestGQA_RatioHandling(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numHeads := 4
	kvHeads := 2
	headDim := 32
	pos := 2
	stride := 8 // ctxLen for scores
	windowSize := 0

	// 1. Prepare Q
	// We want to test that q_head0 and q_head1 both attend to kv_head0,
	// and q_head2 and q_head3 both attend to kv_head1.
	qData := make([]float32, numHeads*headDim)
	for h := 0; h < numHeads; h++ {
		for i := 0; i < headDim; i++ {
			// Make each query head distinct
			qData[h*headDim+i] = float32(h + 1)
		}
	}
	tQ := ctx.NewTensor(1, numHeads*headDim)
	tQ.LoadFrom(qData)

	// 2. Prepare K Cache
	// K Cache is [Tokens, KV_Heads, HeadDim]
	kCacheData := make([]float32, (pos+1)*kvHeads*headDim)
	for p := 0; p <= pos; p++ {
		for kvh := 0; kvh < kvHeads; kvh++ {
			for i := 0; i < headDim; i++ {
				// Make each KV head and position distinct
				kCacheData[p*kvHeads*headDim+kvh*headDim+i] = float32((p + 1) * (kvh + 1))
			}
		}
	}
	tKCache := ctx.NewTensor(pos+1, kvHeads*headDim)
	tKCache.LoadFrom(kCacheData)

	// 3. Prepare Scores Buffer
	tScores := ctx.NewTensorFP32(numHeads, stride)
	tScores.ZeroInit()

	// 4. Run Scores Kernel
	tQ.AttentionScores(tKCache, tScores, pos, numHeads, kvHeads, headDim, stride, windowSize)
	ctx.Synchronize()

	// 5. Verify Scores
	scores := tScores.ToHostF32()
	scale := 1.0 / math.Sqrt(float64(headDim))

	for h := 0; h < numHeads; h++ {
		kvh := h / (numHeads / kvHeads)
		for p := 0; p <= pos; p++ {
			// Expected dot product: Sum_i (Q_h_i * K_p_kvh_i)
			expectedDot := float64(0)
			for i := 0; i < headDim; i++ {
				expectedDot += float64(h+1) * float64((p+1)*(kvh+1))
			}
			expectedScore := float32(expectedDot * scale)

			got := scores[h*stride+p]
			if math.Abs(float64(got-expectedScore)) > 1e-4 {
				t.Errorf("Score mismatch at head %d, pos %d: got %f, want %f (kvh=%d)", h, p, got, expectedScore, kvh)
			}
		}
	}

	// 6. Test Attention Values
	// Prepare V Cache
	vCacheData := make([]float32, (pos+1)*kvHeads*headDim)
	for p := 0; p <= pos; p++ {
		for kvh := 0; kvh < kvHeads; kvh++ {
			for i := 0; i < headDim; i++ {
				vCacheData[p*kvHeads*headDim+kvh*headDim+i] = float32((p + 1) + (kvh + 1) + i)
			}
		}
	}
	tVCache := ctx.NewTensor(pos+1, kvHeads*headDim)
	tVCache.LoadFrom(vCacheData)

	tOut := ctx.NewTensor(1, numHeads*headDim)
	tOut.ZeroInit()

	// Softmax before Values
	tScores.AttSoftmax(pos, numHeads, stride)
	ctx.Synchronize()

	// Run Values Kernel
	tScores.AttValues(tVCache, tOut, pos, numHeads, kvHeads, headDim, stride, windowSize)
	ctx.Synchronize()

	// 7. Verify Values
	gotValues := tOut.ToHost()
	for h := 0; h < numHeads; h++ {
		kvh := h / (numHeads / kvHeads)

		// Expected value: Sum_p (Softmax_hp * V_p_kvh)
		// Softmax values for head h:
		headScores := make([]float64, pos+1)
		sumExp := float64(0)
		for p := 0; p <= pos; p++ {
			// Redo CPU softmax for reference
			expVal := math.Exp(float64(scores[h*stride+p]) - 0) // We don't have max here, but it's small
			headScores[p] = expVal
			sumExp += expVal
		}
		for p := 0; p <= pos; p++ {
			headScores[p] /= sumExp
		}

		for i := 0; i < headDim; i++ {
			expectedVal := float64(0)
			for p := 0; p <= pos; p++ {
				v_val := float64((p + 1) + (kvh + 1) + i)
				expectedVal += headScores[p] * v_val
			}

			got := float64(gotValues[h*headDim+i])
			if math.Abs(got-expectedVal) > 1e-2 { // FP16 precision
				t.Errorf("Value mismatch at head %d, dim %d: got %f, want %f", h, i, got, expectedVal)
				if i > 5 {
					break
				}
			}
		}
	}
}
