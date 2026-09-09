package simd

import (
	"encoding/binary"
	"math"
	"runtime"
	"sync"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

// VNNI dot product function pointers (set from avx512.go when VNNI is available)
var (
	dotQ8_0VNNI func(data []byte, vector []float32, n int) float32
	dotQ4KVNNI  func(data []byte, vector []float32, n int) float32
)

func SoftmaxF32(x []float32) {
	if len(x) == 0 {
		return
	}

	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}

	sum := float32(0.0)
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i] - max)))
		sum += x[i]
	}

	if sum > 0 {
		invSum := float32(1.0) / sum
		for i := range x {
			x[i] *= invSum
		}
	}
}

// MatMul computes A [rowsA x colsA] × B [colsA x colsB] → result [rowsA x colsB].
// Uses parallel outer-row goroutines for matrices larger than 1024 output elements
// and VecDotF32 for SIMD-accelerated inner dot products.
// B is pre-transposed to column-major for cache-friendly column access.
func MatMul(a, b []float32, rowsA, colsA, colsB int) []float32 {
	result := make([]float32, rowsA*colsB)
	if rowsA == 0 || colsA == 0 || colsB == 0 {
		return result
	}

	// Transpose B to column-major: bT[j*colsA+k] = b[k*colsB+j]
	// This makes each column contiguous for VecDotF32.
	bT := make([]float32, colsA*colsB)
	for k := 0; k < colsA; k++ {
		rowOff := k * colsB
		for j := 0; j < colsB; j++ {
			bT[j*colsA+k] = b[rowOff+j]
		}
	}

	workers := runtime.GOMAXPROCS(0)
	if workers > rowsA {
		workers = rowsA
	}
	if workers <= 1 || rowsA*colsB < 1024 {
		for i := 0; i < rowsA; i++ {
			aRow := a[i*colsA : (i+1)*colsA]
			for j := 0; j < colsB; j++ {
				result[i*colsB+j] = VecDotF32(aRow, bT[j*colsA:(j+1)*colsA])
			}
		}
		return result
	}

	rowsPerWorker := (rowsA + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * rowsPerWorker
		end := start + rowsPerWorker
		if start >= rowsA {
			break
		}
		if end > rowsA {
			end = rowsA
		}
		wg.Add(1)
		go func(rStart, rEnd int) {
			defer wg.Done()
			for i := rStart; i < rEnd; i++ {
				aRow := a[i*colsA : (i+1)*colsA]
				for j := 0; j < colsB; j++ {
					result[i*colsB+j] = VecDotF32(aRow, bT[j*colsA:(j+1)*colsA])
				}
			}
		}(start, end)
	}
	wg.Wait()
	return result
}


func MatVecMul(matrix []float32, vector []float32, rows, cols int) []float32 {
	result := make([]float32, rows)
	if rows == 0 || cols == 0 || len(vector) < cols || len(matrix) < rows*cols {
		return result
	}

	workers := runtime.GOMAXPROCS(0)
	if workers > rows {
		workers = rows
	}
	if workers <= 1 || rows < 32 {
		for i := 0; i < rows; i++ {
			result[i] = VecDotF32(matrix[i*cols:(i+1)*cols], vector)
		}
		return result
	}

	rowsPerWorker := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * rowsPerWorker
		end := start + rowsPerWorker
		if start >= rows {
			break
		}
		if end > rows {
			end = rows
		}
		wg.Add(1)
		go func(rStart, rEnd int) {
			defer wg.Done()
			for i := rStart; i < rEnd; i++ {
				result[i] = VecDotF32(matrix[i*cols:(i+1)*cols], vector)
			}
		}(start, end)
	}
	wg.Wait()
	return result
}

func AttentionF32(q, k, v []float32, seqLen, numHeads, headDim int) []float32 {
	scale := 1.0 / math.Sqrt(float64(headDim))
	result := make([]float32, seqLen*headDim)

	if seqLen <= 1 || headDim <= 0 {
		return result
	}

	attnScores := make([]float32, seqLen*seqLen)
	for h := 0; h < numHeads; h++ {
		qHead := q[h*seqLen*headDim : (h+1)*seqLen*headDim]
		kHead := k[h*seqLen*headDim : (h+1)*seqLen*headDim]
		vHead := v[h*seqLen*headDim : (h+1)*seqLen*headDim]
		outHead := result[h*seqLen*headDim : (h+1)*seqLen*headDim]

		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				if j > i {
					attnScores[i*seqLen+j] = float32(-math.Inf(1))
					continue
				}
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += float64(qHead[i*headDim+d]) * float64(kHead[j*headDim+d])
				}
				attnScores[i*seqLen+j] = float32(dot * scale)
			}
		}

		maxScores := make([]float32, seqLen)
		for i := 0; i < seqLen; i++ {
			maxScore := attnScores[i*seqLen]
			for j := 1; j <= i; j++ {
				if attnScores[i*seqLen+j] > maxScore {
					maxScore = attnScores[i*seqLen+j]
				}
			}
			maxScores[i] = maxScore
		}

		expSums := make([]float32, seqLen)
		for i := 0; i < seqLen; i++ {
			var expSum float64
			for j := 0; j <= i; j++ {
				expSum += math.Exp(float64(attnScores[i*seqLen+j]) - float64(maxScores[i]))
			}
			expSums[i] = float32(expSum)
		}

		for i := 0; i < seqLen; i++ {
			if expSums[i] == 0 {
				continue
			}
			for d := 0; d < headDim; d++ {
				var attnSum float64
				for j := 0; j <= i; j++ {
					weight := math.Exp(float64(attnScores[i*seqLen+j]) - float64(maxScores[i]))
					attnSum += weight * float64(vHead[j*headDim+d])
				}
				outHead[i*headDim+d] = float32(attnSum / float64(expSums[i]))
			}
		}
	}

	return result
}

// VecDotF32 computes the dot product of two float32 slices with 8-fold loop unrolling.
func VecDotF32(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i <= n-8; i += 8 {
		s0 += a[i+0] * b[i+0]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
		s4 += a[i+4] * b[i+4]
		s5 += a[i+5] * b[i+5]
		s6 += a[i+6] * b[i+6]
		s7 += a[i+7] * b[i+7]
	}
	sum := ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// VecFMAF32 computes dst[i] += weight * src[i] for float32 slices with 8-fold loop unrolling.
func VecFMAF32(dst, src []float32, weight float32) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	i := 0
	for ; i <= n-8; i += 8 {
		dst[i+0] += weight * src[i+0]
		dst[i+1] += weight * src[i+1]
		dst[i+2] += weight * src[i+2]
		dst[i+3] += weight * src[i+3]
		dst[i+4] += weight * src[i+4]
		dst[i+5] += weight * src[i+5]
		dst[i+6] += weight * src[i+6]
		dst[i+7] += weight * src[i+7]
	}
	for ; i < n; i++ {
		dst[i] += weight * src[i]
	}
}

// Float16ToFloat32 converts a 16-bit half-precision float to IEEE 754 float32.
func Float16ToFloat32(b uint16) float32 {
	sign := uint32(b&0x8000) << 16
	exp := uint32(b&0x7C00) >> 10
	frac := uint32(b&0x03FF) << 13

	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		f := float64(frac) * math.Pow(2, -23)
		if sign != 0 {
			f = -f
		}
		return float32(f * math.Pow(2, -14))
	case 0x1F:
		if frac == 0 {
			if sign != 0 {
				return float32(math.Inf(-1))
			}
			return float32(math.Inf(1))
		}
		return float32(math.NaN())
	default:
		return math.Float32frombits(sign | ((exp + 112) << 23) | frac)
	}
}

// VecDotQ8_0_VNNI computes the dot product of Q8_0 quantized weights and a float32 vector.
// Each 32 elements are packed into 34 bytes (2-byte f16 scale + 32 signed int8s).
// Dispatches to AVX-512 VNNI / AMX if supported, with an unrolled SIMD fallback.
func VecDotQ8_0_VNNI(data []byte, vector []float32) float32 {
	n := len(vector)
	const blockSize = 32
	const blockSizeBytes = 34
	if n == 0 || n%blockSize != 0 {
		return 0
	}
	numBlocks := n / blockSize
	if len(data) < numBlocks*blockSizeBytes {
		return 0
	}

	start := time.Now()
	defer func() {
		metrics.RecordVNNIDotProduct(time.Since(start))
	}()

	if HasAVXVNNI() || HasAMX() {
		metrics.RecordSIMDDispatch("vnni_q8_0")
		if dotQ8_0VNNI != nil {
			return dotQ8_0VNNI(data, vector, n)
		}
	} else {
		metrics.RecordSIMDDispatch("fallback_q8_0")
	}

	var totalSum float32
	for b := 0; b < numBlocks; b++ {
		bOffset := b * blockSizeBytes
		d := Float16ToFloat32(binary.LittleEndian.Uint16(data[bOffset : bOffset+2]))
		vBase := b * blockSize

		var s0, s1, s2, s3 float32
		for j := 0; j < 32; j += 4 {
			s0 += float32(int8(data[bOffset+2+j])) * vector[vBase+j]
			s1 += float32(int8(data[bOffset+3+j])) * vector[vBase+j+1]
			s2 += float32(int8(data[bOffset+4+j])) * vector[vBase+j+2]
			s3 += float32(int8(data[bOffset+5+j])) * vector[vBase+j+3]
		}
		totalSum += d * ((s0 + s1) + (s2 + s3))
	}
	return totalSum
}

// VecDotQ4_K_VNNI computes the dot product of Q4_K quantized weights and a float32 vector.
// Each block of 256 weights occupies 144 bytes with hierarchical super-block scales.
// Dispatches to AVX-512 VNNI / AMX if supported, with an unrolled SIMD fallback.
func VecDotQ4_K_VNNI(data []byte, vector []float32) float32 {
	n := len(vector)
	const blockSize = 256
	const blockSizeBytes = 144
	if n == 0 || n%blockSize != 0 {
		return 0
	}
	numBlocks := n / blockSize
	if len(data) < numBlocks*blockSizeBytes {
		return 0
	}

	start := time.Now()
	defer func() {
		metrics.RecordVNNIDotProduct(time.Since(start))
	}()

	if HasAVXVNNI() || HasAMX() {
		metrics.RecordSIMDDispatch("vnni_q4_k")
		if dotQ4KVNNI != nil {
			return dotQ4KVNNI(data, vector, n)
		}
	} else {
		metrics.RecordSIMDDispatch("fallback_q4_k")
	}

	var totalSum float32
	for i := 0; i < numBlocks; i++ {
		blockOffset := i * blockSizeBytes
		d := Float16ToFloat32(binary.LittleEndian.Uint16(data[blockOffset : blockOffset+2]))
		scales := data[blockOffset+4 : blockOffset+16]
		qs := data[blockOffset+16 : blockOffset+144]

		sc := [8]uint8{
			scales[0] & 63, scales[1] & 63, scales[2] & 63, scales[3] & 63,
			scales[8] & 0x0F, scales[9] & 0x0F, scales[10] & 0x0F, scales[11] & 0x0F,
		}
		m := [8]uint8{
			scales[4] & 63, scales[5] & 63, scales[6] & 63, scales[7] & 63,
			scales[8] >> 4, scales[9] >> 4, scales[10] >> 4, scales[11] >> 4,
		}

		var D [8]float32
		for j := 0; j < 8; j++ {
			D[j] = d * float32(sc[j]) / 225.0
		}

		baseIdx := i * blockSize
		var blockSum float32
		for j := 0; j < 8; j++ {
			step := D[j]
			mj := float32(m[j])
			qsOffset := j * 16
			idxBase := baseIdx + j*32

			var s0, s1 float32
			for k := 0; k < 16; k++ {
				bVal := qs[qsOffset+k]
				s0 += (float32(bVal&0xF) - mj) * vector[idxBase+k]
				s1 += (float32(bVal>>4) - mj) * vector[idxBase+k+16]
			}
			blockSum += step * (s0 + s1)
		}
		totalSum += blockSum
	}
	return totalSum
}

// MatVecMulQ8_0_VNNI computes matrix-vector multiplication for Q8_0 weights.
func MatVecMulQ8_0_VNNI(data []byte, vector []float32, rows, cols int) []float32 {
	result := make([]float32, rows)
	const blockSize = 32
	const blockSizeBytes = 34
	if cols%blockSize != 0 || len(vector) < cols {
		return result
	}
	blocksPerRow := cols / blockSize
	rowBytes := blocksPerRow * blockSizeBytes

	workers := runtime.GOMAXPROCS(0)
	if workers > rows {
		workers = rows
	}
	if workers <= 1 || rows < 16 {
		for r := 0; r < rows; r++ {
			rowSlice := data[r*rowBytes : (r+1)*rowBytes]
			result[r] = VecDotQ8_0_VNNI(rowSlice, vector[:cols])
		}
		return result
	}

	rowsPerWorker := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if startRow >= rows {
			break
		}
		if endRow > rows {
			endRow = rows
		}
		wg.Add(1)
		go func(rStart, rEnd int) {
			defer wg.Done()
			for r := rStart; r < rEnd; r++ {
				rowSlice := data[r*rowBytes : (r+1)*rowBytes]
				result[r] = VecDotQ8_0_VNNI(rowSlice, vector[:cols])
			}
		}(startRow, endRow)
	}
	wg.Wait()
	return result
}

// MatVecMulQ4_K_VNNI computes matrix-vector multiplication for Q4_K weights.
func MatVecMulQ4_K_VNNI(data []byte, vector []float32, rows, cols int) []float32 {
	result := make([]float32, rows)
	const blockSize = 256
	const blockSizeBytes = 144
	if cols%blockSize != 0 || len(vector) < cols {
		return result
	}
	blocksPerRow := cols / blockSize
	rowBytes := blocksPerRow * blockSizeBytes

	workers := runtime.GOMAXPROCS(0)
	if workers > rows {
		workers = rows
	}
	if workers <= 1 || rows < 16 {
		for r := 0; r < rows; r++ {
			rowSlice := data[r*rowBytes : (r+1)*rowBytes]
			result[r] = VecDotQ4_K_VNNI(rowSlice, vector[:cols])
		}
		return result
	}

	rowsPerWorker := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if startRow >= rows {
			break
		}
		if endRow > rows {
			endRow = rows
		}
		wg.Add(1)
		go func(rStart, rEnd int) {
			defer wg.Done()
			for r := rStart; r < rEnd; r++ {
				rowSlice := data[r*rowBytes : (r+1)*rowBytes]
				result[r] = VecDotQ4_K_VNNI(rowSlice, vector[:cols])
			}
		}(startRow, endRow)
	}
	wg.Wait()
	return result
}

