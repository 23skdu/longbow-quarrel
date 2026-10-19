//go:build linux && amd64 && cuda && cgo

package device

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

func float32ToBytes(f32 []float32) []byte {
	buf := make([]byte, len(f32)*4)
	for i, v := range f32 {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

func TestCUDARMSNorm(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	rows, cols := 4, 2048
	input := make([]float32, rows*cols)
	for i := range input {
		input[i] = float32(rand.Float64() * 2)
	}
	weight := make([]float32, cols)
	for i := range weight {
		weight[i] = 1.0 + float32(rand.Float64()*0.1)
	}

	inputTensor, _ := ctx.NewTensorFromData(rows, cols, DataTypeF32, float32ToBytes(input))
	weightTensor, _ := ctx.NewTensorFromData(1, cols, DataTypeF32, float32ToBytes(weight))
	outputTensor, _ := ctx.NewTensor(rows, cols, DataTypeF32)
	hiddenTensor, _ := ctx.NewTensor(rows, cols, DataTypeF32)

	eps := float32(1e-5)
	ctx.FusedRMSNormAdd(inputTensor, hiddenTensor, weightTensor, outputTensor, rows, cols, eps)
	ctx.Synchronize()

	output := outputTensor.ToHostF32()

	if len(output) != len(input) {
		t.Errorf("Output length mismatch: got %d, want %d", len(output), len(input))
	}

	t.Logf("RMSNorm completed - output length: %d", len(output))
}

func TestCUDASwiGLU(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	rows, size := 4, 8192
	gate := make([]float32, rows*size)
	up := make([]float32, rows*size)
	for i := range gate {
		gate[i] = float32(rand.Float64()*2 - 1)
		up[i] = float32(rand.Float64()*2 - 1)
	}

	gateTensor, _ := ctx.NewTensorFromData(rows, size, DataTypeF32, float32ToBytes(gate))
	upTensor, _ := ctx.NewTensorFromData(rows, size, DataTypeF32, float32ToBytes(up))
	downTensor, _ := ctx.NewTensor(rows, size, DataTypeF32)

	ctx.FusedSwiGLU(gateTensor, upTensor, downTensor, rows, size)
	ctx.Synchronize()

	output := downTensor.ToHostF32()

	if len(output) != len(gate) {
		t.Errorf("Output length mismatch: got %d, want %d", len(output), len(gate))
	}

	t.Logf("SwiGLU completed - output length: %d", len(output))
}

func TestCUDALayerScratch(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	qNormDim := 512
	kNormDim := 512
	scratch := ctx.NewLayerScratch(4096, 4096, 14336, 32, 32, 128, 2048, 49152, qNormDim, kNormDim)

	if scratch == nil {
		t.Errorf("Layer scratch is nil")
	}

	defer scratch.Free()
	t.Logf("Layer scratch structure created successfully")
}

func BenchmarkCUDAKernel(b *testing.B) {
	ctx := NewContext()
	defer ctx.Free()

	rows, cols := 32, 4096
	input := make([]float32, rows*cols)
	weight := make([]float32, cols)
	for i := range input {
		input[i] = float32(rand.Float64() * 2)
	}
	for i := range weight {
		weight[i] = 1.0 + float32(rand.Float64()*0.1)
	}

	inputTensor, _ := ctx.NewTensorFromData(rows, cols, DataTypeF32, float32ToBytes(input))
	weightTensor, _ := ctx.NewTensorFromData(1, cols, DataTypeF32, float32ToBytes(weight))
	outputTensor, _ := ctx.NewTensor(rows, cols, DataTypeF32)
	hiddenTensor, _ := ctx.NewTensor(rows, cols, DataTypeF32)

	eps := float32(1e-5)

	b.ResetTimer()
	for b.Loop() {
		ctx.FusedRMSNormAdd(inputTensor, hiddenTensor, weightTensor, outputTensor, rows, cols, eps)
		ctx.Synchronize()
	}
}

func TestCUDA_FusedQKVRope(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	batch := 1
	dim := 256
	heads := 4
	kvHeads := 2
	headDim := 64
	qDim := heads * headDim
	kvDim := kvHeads * headDim
	rotaryDim := 64
	theta := float32(10000.0)
	pos := 7
	posIds := []int{pos}

	inputData := make([]float32, batch*dim)
	for i := range inputData {
		inputData[i] = float32(rand.Float64()*2.0 - 1.0)
	}

	qWData := make([]float32, qDim*dim)
	for i := range qWData {
		qWData[i] = float32(rand.Float64()*0.1 - 0.05)
	}
	kWData := make([]float32, kvDim*dim)
	for i := range kWData {
		kWData[i] = float32(rand.Float64()*0.1 - 0.05)
	}
	vWData := make([]float32, kvDim*dim)
	for i := range vWData {
		vWData[i] = float32(rand.Float64()*0.1 - 0.05)
	}

	inputTensor, _ := ctx.NewTensorFromData(batch, dim, DataTypeF32, float32ToBytes(inputData))
	defer inputTensor.Free()
	qWTensor, _ := ctx.NewTensorFromData(qDim, dim, DataTypeF32, float32ToBytes(qWData))
	defer qWTensor.Free()
	kWTensor, _ := ctx.NewTensorFromData(kvDim, dim, DataTypeF32, float32ToBytes(kWData))
	defer kWTensor.Free()
	vWTensor, _ := ctx.NewTensorFromData(kvDim, dim, DataTypeF32, float32ToBytes(vWData))
	defer vWTensor.Free()

	// 1. Run FusedQKVRope
	q, k, v, err := ctx.FusedQKVRope(inputTensor, qWTensor, kWTensor, vWTensor, posIds, batch, dim, heads, kvHeads, headDim, rotaryDim, theta)
	if err != nil {
		t.Fatalf("FusedQKVRope failed: %v", err)
	}
	ctx.Synchronize()

	qOut := q.ToHostF32()
	kOut := k.ToHostF32()
	vOut := v.ToHostF32()
	t.Logf("qOut[0..4]=%v", qOut[:4])
	t.Logf("kOut[0..4]=%v", kOut[:4])
	t.Logf("vOut[0..4]=%v", vOut[:4])
	q.ReturnToPool()
	k.ReturnToPool()
	v.ReturnToPool()

	// 2. Compute reference outputs on CPU
	// Check V: simple dot products
	for r := 0; r < kvDim; r++ {
		var expected float32
		for c := 0; c < dim; c++ {
			expected += inputData[c] * vWData[r*dim+c]
		}
		diff := vOut[r] - expected
		if diff < -1e-3 || diff > 1e-3 {
			t.Fatalf("V mismatch at row %d: got %f, want %f", r, vOut[r], expected)
		}
	}

	// Check Q with RoPE
	headDimHalf := headDim / 2
	for h := 0; h < heads; h++ {
		for d := 0; d < headDimHalf; d++ {
			r0 := h*headDim + d
			r1 := h*headDim + d + headDimHalf
			var dot0, dot1 float32
			for c := 0; c < dim; c++ {
				dot0 += inputData[c] * qWData[r0*dim+c]
				dot1 += inputData[c] * qWData[r1*dim+c]
			}
			angle := float64(pos) * (1.0 / math.Pow(float64(theta), float64(2*d)/float64(headDim)))
			cosVal := float32(math.Cos(angle))
			sinVal := float32(math.Sin(angle))

			expectedQ0 := dot0*cosVal - dot1*sinVal
			expectedQ1 := dot0*sinVal + dot1*cosVal

			diff0 := qOut[r0] - expectedQ0
			if diff0 < -1e-3 || diff0 > 1e-3 {
				t.Fatalf("Q mismatch at head %d dim %d: got %f, want %f", h, d, qOut[r0], expectedQ0)
			}
			diff1 := qOut[r1] - expectedQ1
			if diff1 < -1e-3 || diff1 > 1e-3 {
				t.Fatalf("Q mismatch at head %d dim %d: got %f, want %f", h, d+headDimHalf, qOut[r1], expectedQ1)
			}
		}
	}

	// Check K with RoPE
	for h := 0; h < kvHeads; h++ {
		for d := 0; d < headDimHalf; d++ {
			r0 := h*headDim + d
			r1 := h*headDim + d + headDimHalf
			var dot0, dot1 float32
			for c := 0; c < dim; c++ {
				dot0 += inputData[c] * kWData[r0*dim+c]
				dot1 += inputData[c] * kWData[r1*dim+c]
			}
			angle := float64(pos) * (1.0 / math.Pow(float64(theta), float64(2*d)/float64(headDim)))
			cosVal := float32(math.Cos(angle))
			sinVal := float32(math.Sin(angle))

			expectedK0 := dot0*cosVal - dot1*sinVal
			expectedK1 := dot0*sinVal + dot1*cosVal

			diff0 := kOut[r0] - expectedK0
			if diff0 < -1e-3 || diff0 > 1e-3 {
				t.Fatalf("K mismatch at kvHead %d dim %d: got %f, want %f", h, d, kOut[r0], expectedK0)
			}
			diff1 := kOut[r1] - expectedK1
			if diff1 < -1e-3 || diff1 > 1e-3 {
				t.Fatalf("K mismatch at kvHead %d dim %d: got %f, want %f", h, d+headDimHalf, kOut[r1], expectedK1)
			}
		}
	}

	t.Logf("FusedQKVRope passed numerical validation against CPU reference")
}

func TestCUDA_FusedQKVRope_Partial(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	batch := 1
	dim := 128
	heads := 2
	kvHeads := 2
	headDim := 64
	rotaryDim := 32 // Only first 32 dims rotated (16 pairs rotated, 16 pairs unrotated)
	qDim := heads * headDim
	kvDim := kvHeads * headDim
	theta := float32(10000.0)
	pos := 12
	posIds := []int{pos}

	inputData := make([]float32, batch*dim)
	for i := range inputData {
		inputData[i] = float32(rand.Float64()*2.0 - 1.0)
	}

	qWData := make([]float32, qDim*dim)
	for i := range qWData {
		qWData[i] = float32(rand.Float64()*0.1 - 0.05)
	}
	kWData := make([]float32, kvDim*dim)
	for i := range kWData {
		kWData[i] = float32(rand.Float64()*0.1 - 0.05)
	}
	vWData := make([]float32, kvDim*dim)
	for i := range vWData {
		vWData[i] = float32(rand.Float64()*0.1 - 0.05)
	}

	inputTensor, _ := ctx.NewTensorFromData(batch, dim, DataTypeF32, float32ToBytes(inputData))
	defer inputTensor.Free()
	qWTensor, _ := ctx.NewTensorFromData(qDim, dim, DataTypeF32, float32ToBytes(qWData))
	defer qWTensor.Free()
	kWTensor, _ := ctx.NewTensorFromData(kvDim, dim, DataTypeF32, float32ToBytes(kWData))
	defer kWTensor.Free()
	vWTensor, _ := ctx.NewTensorFromData(kvDim, dim, DataTypeF32, float32ToBytes(vWData))
	defer vWTensor.Free()

	q, k, v, err := ctx.FusedQKVRope(inputTensor, qWTensor, kWTensor, vWTensor, posIds, batch, dim, heads, kvHeads, headDim, rotaryDim, theta)
	if err != nil {
		t.Fatalf("FusedQKVRope failed: %v", err)
	}
	ctx.Synchronize()

	qOut := q.ToHostF32()
	kOut := k.ToHostF32()
	_ = v.ToHostF32()
	q.ReturnToPool()
	k.ReturnToPool()
	v.ReturnToPool()

	headDimHalf := headDim / 2
	rotaryDimHalf := rotaryDim / 2
	for h := 0; h < heads; h++ {
		for d := 0; d < headDimHalf; d++ {
			r0 := h*headDim + d
			r1 := h*headDim + d + headDimHalf
			var dot0, dot1 float32
			for c := 0; c < dim; c++ {
				dot0 += inputData[c] * qWData[r0*dim+c]
				dot1 += inputData[c] * qWData[r1*dim+c]
			}

			if d < rotaryDimHalf {
				angle := float64(pos) * (1.0 / math.Pow(float64(theta), float64(2*d)/float64(rotaryDim)))
				cosVal := float32(math.Cos(angle))
				sinVal := float32(math.Sin(angle))
				expected0 := dot0*cosVal - dot1*sinVal
				expected1 := dot0*sinVal + dot1*cosVal
				diff0 := qOut[r0] - expected0
				if diff0 < -1e-3 || diff0 > 1e-3 {
					t.Fatalf("Partial RoPE rotated mismatch at head %d dim %d: got %f, want %f", h, d, qOut[r0], expected0)
				}
				diff1 := qOut[r1] - expected1
				if diff1 < -1e-3 || diff1 > 1e-3 {
					t.Fatalf("Partial RoPE rotated mismatch at head %d dim %d: got %f, want %f", h, d+headDimHalf, qOut[r1], expected1)
				}
			} else {
				// Unrotated
				diff0 := qOut[r0] - dot0
				if diff0 < -1e-3 || diff0 > 1e-3 {
					t.Fatalf("Partial RoPE unrotated mismatch at head %d dim %d: got %f, want %f", h, d, qOut[r0], dot0)
				}
				diff1 := qOut[r1] - dot1
				if diff1 < -1e-3 || diff1 > 1e-3 {
					t.Fatalf("Partial RoPE unrotated mismatch at head %d dim %d: got %f, want %f", h, d+headDimHalf, qOut[r1], dot1)
				}
			}
		}
	}

	for h := 0; h < kvHeads; h++ {
		for d := 0; d < headDimHalf; d++ {
			r0 := h*headDim + d
			r1 := h*headDim + d + headDimHalf
			var dot0, dot1 float32
			for c := 0; c < dim; c++ {
				dot0 += inputData[c] * kWData[r0*dim+c]
				dot1 += inputData[c] * kWData[r1*dim+c]
			}

			if d < rotaryDimHalf {
				angle := float64(pos) * (1.0 / math.Pow(float64(theta), float64(2*d)/float64(rotaryDim)))
				cosVal := float32(math.Cos(angle))
				sinVal := float32(math.Sin(angle))
				expected0 := dot0*cosVal - dot1*sinVal
				expected1 := dot0*sinVal + dot1*cosVal
				diff0 := kOut[r0] - expected0
				if diff0 < -1e-3 || diff0 > 1e-3 {
					t.Fatalf("Partial RoPE K rotated mismatch at kvHead %d dim %d: got %f, want %f", h, d, kOut[r0], expected0)
				}
				diff1 := kOut[r1] - expected1
				if diff1 < -1e-3 || diff1 > 1e-3 {
					t.Fatalf("Partial RoPE K rotated mismatch at kvHead %d dim %d: got %f, want %f", h, d+headDimHalf, kOut[r1], expected1)
				}
			} else {
				// Unrotated
				diff0 := kOut[r0] - dot0
				if diff0 < -1e-3 || diff0 > 1e-3 {
					t.Fatalf("Partial RoPE K unrotated mismatch at kvHead %d dim %d: got %f, want %f", h, d, kOut[r0], dot0)
				}
				diff1 := kOut[r1] - dot1
				if diff1 < -1e-3 || diff1 > 1e-3 {
					t.Fatalf("Partial RoPE K unrotated mismatch at kvHead %d dim %d: got %f, want %f", h, d+headDimHalf, kOut[r1], dot1)
				}
			}
		}
	}

	t.Logf("Partial RoPE verified successfully")
}

func TestCUDA_QKNormRMSNorm(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numTokens := 3
	qHeads := 8
	kHeads := 2
	headDim := 128
	eps := float32(1e-6)

	qLen := numTokens * qHeads * headDim
	kLen := numTokens * kHeads * headDim

	qHost := make([]float32, qLen)
	kHost := make([]float32, kLen)
	for i := range qHost {
		qHost[i] = float32(rand.Float64()*2.0 - 1.0)
	}
	for i := range kHost {
		kHost[i] = float32(rand.Float64()*2.0 - 1.0)
	}

	qWeightHost := make([]float32, headDim)
	kWeightHost := make([]float32, headDim)
	for i := range qWeightHost {
		qWeightHost[i] = float32(rand.Float64()*1.5 + 0.5)
		kWeightHost[i] = float32(rand.Float64()*1.5 + 0.5)
	}

	qTensor, _ := ctx.NewTensorFromData(1, qLen, DataTypeF32, float32ToBytes(qHost))
	defer qTensor.Free()
	kTensor, _ := ctx.NewTensorFromData(1, kLen, DataTypeF32, float32ToBytes(kHost))
	defer kTensor.Free()
	qWTensor, _ := ctx.NewTensorFromData(1, headDim, DataTypeF32, float32ToBytes(qWeightHost))
	defer qWTensor.Free()
	kWTensor, _ := ctx.NewTensorFromData(1, headDim, DataTypeF32, float32ToBytes(kWeightHost))
	defer kWTensor.Free()

	ctx.QKNormRMSNorm(qTensor, kTensor, qWTensor, kWTensor, numTokens, qHeads, kHeads, headDim, eps)
	ctx.Synchronize()

	qResult := qTensor.ToHostF32()
	kResult := kTensor.ToHostF32()

	// CPU reference verification for Q
	for tok := 0; tok < numTokens; tok++ {
		for h := 0; h < qHeads; h++ {
			offset := (tok*qHeads + h) * headDim
			var sumSq float64
			for d := 0; d < headDim; d++ {
				v := float64(qHost[offset+d])
				sumSq += v * v
			}
			invRms := float32(1.0 / math.Sqrt(sumSq/float64(headDim)+float64(eps)))
			for d := 0; d < headDim; d++ {
				expected := qHost[offset+d] * invRms * qWeightHost[d]
				actual := qResult[offset+d]
				diff := float32(math.Abs(float64(actual - expected)))
				if diff > 1e-4 {
					t.Fatalf("Q Norm mismatch at tok=%d, head=%d, dim=%d: got %f, want %f (diff %f)",
						tok, h, d, actual, expected, diff)
				}
			}
		}
	}

	// CPU reference verification for K
	for tok := 0; tok < numTokens; tok++ {
		for h := 0; h < kHeads; h++ {
			offset := (tok*kHeads + h) * headDim
			var sumSq float64
			for d := 0; d < headDim; d++ {
				v := float64(kHost[offset+d])
				sumSq += v * v
			}
			invRms := float32(1.0 / math.Sqrt(sumSq/float64(headDim)+float64(eps)))
			for d := 0; d < headDim; d++ {
				expected := kHost[offset+d] * invRms * kWeightHost[d]
				actual := kResult[offset+d]
				diff := float32(math.Abs(float64(actual - expected)))
				if diff > 1e-4 {
					t.Fatalf("K Norm mismatch at tok=%d, head=%d, dim=%d: got %f, want %f (diff %f)",
						tok, h, d, actual, expected, diff)
				}
			}
		}
	}

	t.Logf("QKNormRMSNorm GPU kernel successfully verified against CPU reference")
}

func TestCUDA_AttentionPagedSplitK(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numTokens := 1
	heads := 4
	kvHeads := 2
	headDim := 64
	blockSize := 16
	seqLen := 256
	maxBlocks := (seqLen + blockSize - 1) / blockSize

	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	qData := make([]float32, numTokens*heads*headDim)
	for i := range qData {
		qData[i] = float32(rand.Float64()*0.4 - 0.2)
	}

	kvDim := kvHeads * headDim
	poolSize := maxBlocks * blockSize * kvDim
	kPoolFP16 := make([]uint16, poolSize)
	vPoolFP16 := make([]uint16, poolSize)
	for i := range kPoolFP16 {
		kPoolFP16[i] = Float32ToFloat16(float32(rand.Float64()*0.4 - 0.2))
		vPoolFP16[i] = Float32ToFloat16(float32(rand.Float64()*0.4 - 0.2))
	}

	blockTable := make([]int32, maxBlocks)
	for b := 0; b < maxBlocks; b++ {
		blockTable[b] = int32(b)
	}

	tokenPositions := []int32{int32(seqLen - 1)}
	tokenToSeq := []int32{0}

	qTensor, _ := ctx.NewTensorFromData(1, len(qData), DataTypeF32, float32ToBytes(qData))
	defer qTensor.Free()

	kBytes := make([]byte, len(kPoolFP16)*2)
	vBytes := make([]byte, len(vPoolFP16)*2)
	for i, v := range kPoolFP16 {
		binary.LittleEndian.PutUint16(kBytes[i*2:], v)
	}
	for i, v := range vPoolFP16 {
		binary.LittleEndian.PutUint16(vBytes[i*2:], v)
	}

	kPoolTensor, _ := ctx.NewTensorFromData(1, poolSize, DataTypeF16, kBytes)
	defer kPoolTensor.Free()
	vPoolTensor, _ := ctx.NewTensorFromData(1, poolSize, DataTypeF16, vBytes)
	defer vPoolTensor.Free()

	posTensor := ctx.NewTensorI32(1, numTokens)
	defer posTensor.Free()
	_ = posTensor.LoadFrom(tokenPositions)

	tokSeqTensor := ctx.NewTensorI32(1, numTokens)
	defer tokSeqTensor.Free()
	_ = tokSeqTensor.LoadFrom(tokenToSeq)

	blockTableTensor := ctx.NewTensorI32(1, maxBlocks)
	defer blockTableTensor.Free()
	_ = blockTableTensor.LoadFrom(blockTable)

	refOut := ctx.NewTensorFP32(numTokens, heads*headDim)
	defer refOut.Free()

	splitOut := ctx.NewTensorFP32(numTokens, heads*headDim)
	defer splitOut.Free()

	// 1. Compute reference attention via AttentionPagedBatch
	ctx.AttentionPagedBatch(qTensor, kPoolTensor, vPoolTensor, refOut, posTensor, blockTableTensor,
		maxBlocks, heads, kvHeads, headDim, blockSize, tokSeqTensor, 1)
	ctx.Synchronize()

	// 2. Compute attention via Split-K FlashDecoding (splitK = 4)
	ctx.AttentionPagedSplitK(qTensor, kPoolTensor, vPoolTensor, splitOut, posTensor, blockTableTensor, tokSeqTensor,
		maxBlocks, heads, kvHeads, headDim, blockSize, numTokens, scale, 4)
	ctx.Synchronize()

	refHost := refOut.ToHostF32()
	splitHost := splitOut.ToHostF32()

	var maxDiff float32
	for i := range refHost {
		diff := float32(math.Abs(float64(refHost[i] - splitHost[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-3 {
			t.Fatalf("Split-K mismatch at idx %d: ref=%f, split=%f (diff=%f)", i, refHost[i], splitHost[i], diff)
		}
	}

	t.Logf("Split-K FlashDecoding verified successfully against reference (max diff: %e)", maxDiff)
}


