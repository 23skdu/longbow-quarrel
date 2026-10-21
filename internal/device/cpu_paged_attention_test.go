//go:build !metal && !cuda

package device

import (
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/mathx"
	"github.com/23skdu/longbow-quarrel/internal/simd"
)

// TestCPU_AttentionPagedBatch exercises Context.AttentionPagedBatch and
// Context.AttentionPagedSplitK against an independent CPU reference computation.
func TestCPU_AttentionPagedBatch(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	const (
		batchSize       = 2
		heads           = 4
		kvHeads         = 2
		headDim         = 16
		blockSize       = 8
		maxBlocksPerSeq = 4
		totalBlocks     = 8
	)

	// q: [batchSize, heads, headDim]
	q := ctx.NewTensorFP32(batchSize, heads*headDim)
	for i := range q.data {
		q.data[i] = float32(math.Sin(float64(i+1) * 0.1))
	}

	// kCache, vCache: [totalBlocks * blockSize * kvHeads * headDim]
	cacheSize := totalBlocks * blockSize * kvHeads * headDim
	kCache := ctx.NewTensorFP32(1, cacheSize)
	vCache := ctx.NewTensorFP32(1, cacheSize)
	for i := range kCache.data {
		kCache.data[i] = float32(math.Cos(float64(i+1) * 0.05))
		vCache.data[i] = float32(math.Sin(float64(i+1) * 0.03))
	}

	output := ctx.NewTensorFP32(batchSize, heads*headDim)
	output.ZeroInit()

	// Positions for each batch item: seq 0 at pos 5, seq 1 at pos 12
	positions := []int{5, 12}
	tokenPositions := ctx.NewTensor(1, batchSize)
	tokenPositions.rawData = make([]byte, batchSize*4)
	for b, p := range positions {
		setFloat32(tokenPositions.rawData[b*4:], float32(p))
	}

	// Block tables mapping logicalBlockIdx -> physicalBlockID
	// seq 0: pos 5 -> uses block 2 (logical 0)
	// seq 1: pos 12 -> uses block 5 (logical 0, pos 0..7) and block 3 (logical 1, pos 8..12)
	blockTables := ctx.NewTensor(batchSize, maxBlocksPerSeq)
	blockTables.rawData = make([]byte, batchSize*maxBlocksPerSeq*4)
	mapping := [batchSize][maxBlocksPerSeq]int{
		{2, 0, 0, 0},
		{5, 3, 0, 0},
	}
	for b := 0; b < batchSize; b++ {
		for l := 0; l < maxBlocksPerSeq; l++ {
			setFloat32(blockTables.rawData[(b*maxBlocksPerSeq+l)*4:], float32(mapping[b][l]))
		}
	}

	// Execute AttentionPagedBatch
	ctx.AttentionPagedBatch(
		q, kCache, vCache, output,
		tokenPositions, blockTables,
		maxBlocksPerSeq, heads, kvHeads, headDim, blockSize,
		nil, batchSize,
	)

	// Independently compute reference
	scale := float32(1.0 / mathx.Sqrt(float64(headDim)))
	expectedOut := make([]float32, batchSize*heads*headDim)

	for b := 0; b < batchSize; b++ {
		pos := positions[b]
		for h := 0; h < heads; h++ {
			qOff := (b*heads + h) * headDim
			qHead := q.data[qOff : qOff+headDim]

			scores := make([]float32, pos+1)
			for p := 0; p <= pos; p++ {
				logicalBlockIdx := p / blockSize
				blockOffset := p % blockSize
				pBlockID := mapping[b][logicalBlockIdx]

				kOff := ((pBlockID*blockSize+blockOffset)*kvHeads + (h % kvHeads)) * headDim
				kHead := kCache.data[kOff : kOff+headDim]

				var dot float32
				for i := 0; i < headDim; i++ {
					dot += qHead[i] * kHead[i]
				}
				scores[p] = dot * scale
			}

			simd.SoftmaxAVX2(scores)

			outOff := (b*heads + h) * headDim
			outHead := expectedOut[outOff : outOff+headDim]
			for p := 0; p <= pos; p++ {
				logicalBlockIdx := p / blockSize
				blockOffset := p % blockSize
				pBlockID := mapping[b][logicalBlockIdx]

				vOff := ((pBlockID*blockSize+blockOffset)*kvHeads + (h % kvHeads)) * headDim
				vHead := vCache.data[vOff : vOff+headDim]

				s := scores[p]
				for i := 0; i < headDim; i++ {
					outHead[i] += s * vHead[i]
				}
			}
		}
	}

	// Compare actual output against reference
	var maxDiff float32
	for i := range output.data {
		diff := float32(math.Abs(float64(output.data[i] - expectedOut[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	if maxDiff > 1e-5 {
		t.Errorf("AttentionPagedBatch max diff %g exceeds 1e-5", maxDiff)
	}

	// Verify AttentionPagedSplitK delegation does not fail and matches output
	outputSplitK := ctx.NewTensorFP32(batchSize, heads*headDim)
	outputSplitK.ZeroInit()
	ctx.AttentionPagedSplitK(
		q, kCache, vCache, outputSplitK,
		tokenPositions, blockTables, nil,
		maxBlocksPerSeq, heads, kvHeads, headDim, blockSize,
		batchSize, scale, 2,
	)

	for i := range outputSplitK.data {
		if outputSplitK.data[i] != output.data[i] {
			t.Errorf("AttentionPagedSplitK mismatch at %d: %g != %g", i, outputSplitK.data[i], output.data[i])
			break
		}
	}
}
