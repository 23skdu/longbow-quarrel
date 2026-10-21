//go:build linux && amd64 && cuda && cgo

package device

import (
	"math"
	"testing"
)

// TestCUDAQuantizedKVCacheMatchesTheF16Path checks that the quantized KV
// cache is the same cache, in the sense that the attention over it gives
// the same answer as the attention over the F16 pool.
//
// The quantized pool holds one byte per element and a scale per slot. It
// used to be allocated with four bytes per element, filled with F16 values
// by the ordinary store, and read back as int8 by the quantized attention
// kernel: the three never agreed, and the feature was unreachable because
// nothing ever set the precision.
func TestCUDAQuantizedKVCacheMatchesTheF16Path(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	const (
		kvDim     = 64
		numTokens = 8
		capacity  = 64
		heads     = 2
		headDim   = 32
		blockSize = 16
	)

	for _, tc := range []struct {
		name  string
		isFP8 bool
		tol   float64
	}{
		{name: "int8", isFP8: false, tol: 0.05},
		{name: "fp8", isFP8: true, tol: 0.25},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The F16 pool and the quantized pool hold the same data.
			f16Pool := ctx.NewTensorWithType(capacity, kvDim, DataTypeF16)
			if f16Pool == nil {
				t.Fatal("failed to create the F16 pool")
			}
			defer f16Pool.Free()
			f16VPool := ctx.NewTensorWithType(capacity, kvDim, DataTypeF16)
			if f16VPool == nil {
				t.Fatal("failed to create the F16 value pool")
			}
			defer f16VPool.Free()

			quantPool := ctx.NewTensorWithType(capacity, kvDim, DataTypeINT8)
			if quantPool == nil {
				t.Fatal("failed to create the quantized pool")
			}
			defer quantPool.Free()
			quantVPool := ctx.NewTensorWithType(capacity, kvDim, DataTypeINT8)
			if quantVPool == nil {
				t.Fatal("failed to create the quantized value pool")
			}
			defer quantVPool.Free()

			// The scale of every slot: one float per slot.
			scales := ctx.NewTensorFP32(1, capacity)
			if scales == nil {
				t.Fatal("failed to create the scale tensor")
			}
			defer scales.Free()
			vScales := ctx.NewTensorFP32(1, capacity)
			if vScales == nil {
				t.Fatal("failed to create the value scale tensor")
			}
			defer vScales.Free()

			// The data: a smooth, non-degenerate pattern so that the
			// quantization error shows up rather than cancels out.
			kData := make([]float32, numTokens*kvDim)
			vData := make([]float32, numTokens*kvDim)
			for i := range kData {
				kData[i] = float32(math.Sin(float64(i)*0.37)) * 2
				vData[i] = float32(math.Cos(float64(i)*0.21)) * 2
			}
			kTensor := ctx.NewTensorFP32(numTokens, kvDim)
			if kTensor == nil {
				t.Fatal("failed to create the key tensor")
			}
			defer kTensor.Free()
			if err := kTensor.LoadFrom(kData); err != nil {
				t.Fatalf("failed to load the keys: %v", err)
			}
			vTensor := ctx.NewTensorFP32(numTokens, kvDim)
			if vTensor == nil {
				t.Fatal("failed to create the value tensor")
			}
			defer vTensor.Free()
			if err := vTensor.LoadFrom(vData); err != nil {
				t.Fatalf("failed to load the values: %v", err)
			}

			// Every token goes to its own slot.
			positions := make([]float32, numTokens)
			for i := range positions {
				positions[i] = float32(i)
			}
			physTensor := ctx.NewTensorFP32(1, numTokens)
			if physTensor == nil {
				t.Fatal("failed to create the position tensor")
			}
			defer physTensor.Free()
			if err := physTensor.LoadFrom(positions); err != nil {
				t.Fatalf("failed to load the positions: %v", err)
			}

			// The F16 store.
			ctx.StoreKVPagedBatch(kTensor, vTensor, f16Pool, f16VPool, physTensor, kvDim, numTokens)

			// The quantized store.
			if err := ctx.StoreKVPagedBatchQuantized(kTensor, vTensor, quantPool, quantVPool,
				scales, vScales, physTensor, kvDim, numTokens, tc.isFP8); err != nil {
				t.Fatalf("the quantized store failed: %v", err)
			}
			ctx.Synchronize()

			// The quantized pool holds one byte per element: half the size
			// of the F16 pool it replaces.
			if quantPool.sizeBytes != capacity*kvDim {
				t.Errorf("the quantized pool holds %d bytes, want %d", quantPool.sizeBytes, capacity*kvDim)
			}

			// The scale of a written slot is the amax of the row over
			// the largest code, and not the unit default.
			hostScales := scales.ToHostF32()
			if len(hostScales) < capacity {
				t.Fatalf("the scale tensor holds %d values, want %d", len(hostScales), capacity)
			}
			maxCode := float32(127)
			if tc.isFP8 {
				maxCode = 448
			}
			for token := range positions {
				want := float32(0)
				for d := 0; d < kvDim; d++ {
					a := math.Abs(float64(kData[token*kvDim+d]))
					b := math.Abs(float64(vData[token*kvDim+d]))
					if a > float64(want) {
						want = float32(a)
					}
					if b > float64(want) {
						want = float32(b)
					}
				}
				want /= maxCode
				got := hostScales[token]
				if math.Abs(float64(got-want)) > 0.01*float64(want) {
					t.Errorf("the scale of the slot %d is %v, want %v", token, got, want)
				}
			}

			// The same query against both pools.
			qData := make([]float32, numTokens*heads*headDim)
			for i := range qData {
				qData[i] = float32(math.Sin(float64(i) * 0.11))
			}
			q := ctx.NewTensorFP32(numTokens, heads*headDim)
			if q == nil {
				t.Fatal("failed to create the query tensor")
			}
			defer q.Free()
			if err := q.LoadFrom(qData); err != nil {
				t.Fatalf("failed to load the query: %v", err)
			}

			// The token positions and the block table: one sequence, one
			// block, numTokens tokens.
			tokenPos := ctx.NewTensorI32(1, numTokens)
			if tokenPos == nil {
				t.Fatal("failed to create the token position tensor")
			}
			defer tokenPos.Free()
			positions32 := make([]int32, numTokens)
			for i := range positions32 {
				positions32[i] = int32(i)
			}
			if err := tokenPos.LoadFrom(positions32); err != nil {
				t.Fatalf("failed to load the token positions: %v", err)
			}

			tokenToSeq := ctx.NewTensorI32(1, numTokens)
			if tokenToSeq == nil {
				t.Fatal("failed to create the token-to-sequence tensor")
			}
			defer tokenToSeq.Free()
			seqs := make([]int32, numTokens)
			for i := range seqs {
				seqs[i] = 0
			}
			if err := tokenToSeq.LoadFrom(seqs); err != nil {
				t.Fatalf("failed to load the token-to-sequence map: %v", err)
			}

			maxBlocks := 1
			blockTable := ctx.NewTensorI32(1, maxBlocks)
			if blockTable == nil {
				t.Fatal("failed to create the block table")
			}
			defer blockTable.Free()
			if err := blockTable.LoadFrom([]int32{0}); err != nil {
				t.Fatalf("failed to load the block table: %v", err)
			}

			f16Out := ctx.NewTensorFP32(numTokens, heads*headDim)
			if f16Out == nil {
				t.Fatal("failed to create the F16 output tensor")
			}
			defer f16Out.Free()
			quantOut := ctx.NewTensorFP32(numTokens, heads*headDim)
			if quantOut == nil {
				t.Fatal("failed to create the quantized output tensor")
			}
			defer quantOut.Free()

			attentionScale := float32(1.0 / math.Sqrt(float64(headDim)))

			ctx.AttentionPagedBatch(q, f16Pool, f16VPool, f16Out, tokenPos, blockTable,
				maxBlocks, heads, heads, headDim, blockSize, tokenToSeq, numTokens)
			ctx.PagedAttentionQuantized(q, quantPool, quantVPool, scales, vScales, quantOut,
				tokenPos, blockTable, tokenToSeq, maxBlocks, heads, heads, headDim, blockSize,
				numTokens, attentionScale, tc.isFP8)
			ctx.Synchronize()

			f16Host := f16Out.ToHostF32()
			quantHost := quantOut.ToHostF32()
			if len(f16Host) != len(quantHost) {
				t.Fatalf("the two outputs have different sizes: %d and %d", len(f16Host), len(quantHost))
			}

			// The comparison is only meaningful if the attention actually
			// produced something.
			nonZero := 0
			for i := range f16Host {
				if math.Abs(float64(f16Host[i])) > 1e-6 {
					nonZero++
				}
			}
			if nonZero == 0 {
				t.Fatal("the F16 attention produced nothing, so the comparison would pass either way")
			}

			maxDiff := 0.0
			for i := range f16Host {
				if math.IsNaN(float64(quantHost[i])) {
					t.Fatalf("the quantized output holds a NaN at %d", i)
				}
				diff := math.Abs(float64(f16Host[i] - quantHost[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
				if diff > tc.tol {
					t.Fatalf("the outputs differ at %d: the F16 pool gives %v, the %s pool gives %v",
						i, f16Host[i], tc.name, quantHost[i])
				}
			}
			t.Logf("the %s pool differs from the F16 pool by at most %v", tc.name, maxDiff)
		})
	}
}
