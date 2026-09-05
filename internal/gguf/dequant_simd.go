package gguf

import (
	"encoding/binary"
	"runtime"
	"sync"
)

// DequantizeQ4K_SIMD dequantizes Q4_K data with optimized unrolling and parallel workers.
func DequantizeQ4K_SIMD(data []byte, numElements int) []float32 {
	if numElements%BlockSizeQ4K != 0 {
		panic("DequantizeQ4K_SIMD: numElements must be multiple of 256")
	}

	numBlocks := numElements / BlockSizeQ4K
	out := make([]float32, numElements)
	const blockSizeBytes = 144

	if numBlocks < 16 {
		dequantizeQ4KBlocks(data, out, 0, numBlocks)
		return out
	}

	workers := runtime.GOMAXPROCS(0)
	if workers > numBlocks {
		workers = numBlocks
	}

	blocksPerWorker := (numBlocks + workers - 1) / workers
	var wg sync.WaitGroup

	for w := 0; w < workers; w++ {
		startBlock := w * blocksPerWorker
		endBlock := startBlock + blocksPerWorker
		if startBlock >= numBlocks {
			break
		}
		if endBlock > numBlocks {
			endBlock = numBlocks
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			dequantizeQ4KBlocks(data, out, start, end)
		}(startBlock, endBlock)
	}

	wg.Wait()
	return out
}

func dequantizeQ4KBlocks(data []byte, out []float32, startBlock, endBlock int) {
	const blockSizeBytes = 144

	for i := startBlock; i < endBlock; i++ {
		blockOffset := i * blockSizeBytes
		if blockOffset+blockSizeBytes > len(data) {
			break
		}
		blockData := data[blockOffset : blockOffset+blockSizeBytes]

		d := Float16ToFloat32(binary.LittleEndian.Uint16(blockData[0:2]))
		scales := blockData[4:16]
		qs := blockData[16:144]

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

		baseIdx := i * BlockSizeQ4K
		for j := 0; j < 8; j++ {
			step := D[j]
			mj := float32(m[j])
			qsOffset := j * 16
			idxBase := baseIdx + j*32

			// 4x unrolled inner loop
			for k := 0; k < 16; k += 4 {
				b0 := qs[qsOffset+k]
				b1 := qs[qsOffset+k+1]
				b2 := qs[qsOffset+k+2]
				b3 := qs[qsOffset+k+3]

				out[idxBase+k] = step * (float32(b0&0xF) - mj)
				out[idxBase+k+1] = step * (float32(b1&0xF) - mj)
				out[idxBase+k+2] = step * (float32(b2&0xF) - mj)
				out[idxBase+k+3] = step * (float32(b3&0xF) - mj)

				out[idxBase+k+16] = step * (float32(b0>>4) - mj)
				out[idxBase+k+17] = step * (float32(b1>>4) - mj)
				out[idxBase+k+18] = step * (float32(b2>>4) - mj)
				out[idxBase+k+19] = step * (float32(b3>>4) - mj)
			}
		}
	}
}

// DequantizeQ6K_SIMD dequantizes Q6_K data with precomputed scaling, loop unrolling and parallel workers.
func DequantizeQ6K_SIMD(data []byte, numElements int) []float32 {
	if numElements%BlockSizeQ6K != 0 {
		panic("DequantizeQ6K_SIMD: numElements must be multiple of 256")
	}

	const blockSizeBytes = 210
	numBlocks := numElements / BlockSizeQ6K
	out := make([]float32, numElements)

	if numBlocks < 16 {
		dequantizeQ6KBlocks(data, out, 0, numBlocks)
		return out
	}

	workers := runtime.GOMAXPROCS(0)
	if workers > numBlocks {
		workers = numBlocks
	}

	blocksPerWorker := (numBlocks + workers - 1) / workers
	var wg sync.WaitGroup

	for w := 0; w < workers; w++ {
		startBlock := w * blocksPerWorker
		endBlock := startBlock + blocksPerWorker
		if startBlock >= numBlocks {
			break
		}
		if endBlock > numBlocks {
			endBlock = numBlocks
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			dequantizeQ6KBlocks(data, out, start, end)
		}(startBlock, endBlock)
	}

	wg.Wait()
	return out
}

func dequantizeQ6KBlocks(data []byte, out []float32, startBlock, endBlock int) {
	const blockSizeBytes = 210

	for i := startBlock; i < endBlock; i++ {
		blockOffset := i * blockSizeBytes
		if blockOffset+blockSizeBytes > len(data) {
			break
		}
		block := data[blockOffset : blockOffset+blockSizeBytes]

		qs := block[0:128]
		qh := block[128:192]
		scales := block[192:208]
		d := Float16ToFloat32(binary.LittleEndian.Uint16(block[208:210]))

		var effScales [16]float32
		for s := 0; s < 16; s++ {
			effScales[s] = d * float32(int8(scales[s])) // #nosec G115
		}

		base := i * 256

		for si := 0; si < 2; si++ {
			scOff := si * 8
			qhOff := si * 32
			n := si * 128

			for l := 0; l < 32; l++ {
				is := l / 16
				s0 := effScales[scOff+is*2+0]
				s1 := effScales[scOff+is*2+1]
				s2 := effScales[scOff+is*2+2]
				s3 := effScales[scOff+is*2+3]

				qhl := qh[l+qhOff]
				qsl0 := qs[l+0]
				qsl32 := qs[l+32]

				q1 := int8((qsl0&0xF)|(((qhl>>0)&3)<<4)) - 32  // #nosec G115
				q2 := int8((qsl32&0xF)|(((qhl>>2)&3)<<4)) - 32 // #nosec G115
				q3 := int8((qsl0>>4)|(((qhl>>4)&3)<<4)) - 32   // #nosec G115
				q4 := int8((qsl32>>4)|(((qhl>>6)&3)<<4)) - 32  // #nosec G115

				yIdx := base + n + l
				out[yIdx+0] = s0 * float32(q1)
				out[yIdx+32] = s1 * float32(q2)
				out[yIdx+64] = s2 * float32(q3)
				out[yIdx+96] = s3 * float32(q4)
			}
		}
	}
}

// MatVecMulQ4_K performs zero-copy matrix-vector multiplication directly on Q4_K quantized weights.
func MatVecMulQ4_K(data []byte, vector []float32, rows, cols int) []float32 {
	const blockSize = 256
	const blockSizeBytes = 144
	blocksPerRow := cols / blockSize
	rowBytes := blocksPerRow * blockSizeBytes

	result := make([]float32, rows)

	workers := runtime.GOMAXPROCS(0)
	if workers > rows {
		workers = rows
	}
	if workers < 1 {
		workers = 1
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
			var blockDeq [blockSize]float32
			for r := rStart; r < rEnd; r++ {
				rowOffset := r * rowBytes
				if rowOffset+rowBytes > len(data) {
					break
				}
				rowSlice := data[rowOffset : rowOffset+rowBytes]

				var sum float32
				for b := 0; b < blocksPerRow; b++ {
					bOffset := b * blockSizeBytes
					bData := rowSlice[bOffset : bOffset+blockSizeBytes]
					dequantizeQ4KBlocks(bData, blockDeq[:], 0, 1)

					vOffset := b * blockSize
					for k := 0; k < blockSize; k++ {
						sum += blockDeq[k] * vector[vOffset+k]
					}
				}
				result[r] = sum
			}
		}(startRow, endRow)
	}

	wg.Wait()
	return result
}

// MatVecMulQ6_K performs zero-copy matrix-vector multiplication directly on Q6_K quantized weights.
func MatVecMulQ6_K(data []byte, vector []float32, rows, cols int) []float32 {
	const blockSize = 256
	const blockSizeBytes = 210
	blocksPerRow := cols / blockSize
	rowBytes := blocksPerRow * blockSizeBytes

	result := make([]float32, rows)

	workers := runtime.GOMAXPROCS(0)
	if workers > rows {
		workers = rows
	}
	if workers < 1 {
		workers = 1
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
			var blockDeq [blockSize]float32
			for r := rStart; r < rEnd; r++ {
				rowOffset := r * rowBytes
				if rowOffset+rowBytes > len(data) {
					break
				}
				rowSlice := data[rowOffset : rowOffset+rowBytes]

				var sum float32
				for b := 0; b < blocksPerRow; b++ {
					bOffset := b * blockSizeBytes
					bData := rowSlice[bOffset : bOffset+blockSizeBytes]
					dequantizeQ6KBlocks(bData, blockDeq[:], 0, 1)

					vOffset := b * blockSize
					for k := 0; k < blockSize; k++ {
						sum += blockDeq[k] * vector[vOffset+k]
					}
				}
				result[r] = sum
			}
		}(startRow, endRow)
	}

	wg.Wait()
	return result
}

// matVecMulGeneric is a parallel worker-pool template for zero-copy MatVec
// on any block-based quantized format. dequantFn dequantizes one row of rawBytes
// into dst (pre-allocated to blockSize). rows x cols must be exact.
func matVecMulGeneric(data []byte, vector []float32, rows, cols int,
	blockSize, blockSizeBytes int,
	dequantFn func(block []byte, dst []float32)) []float32 {
	result := make([]float32, rows)
	if cols == 0 || blockSize == 0 || cols%blockSize != 0 {
		return result
	}
	blocksPerRow := cols / blockSize
	rowBytes := blocksPerRow * blockSizeBytes
	if len(data) < rows*rowBytes {
		return result
	}

	workers := runtime.GOMAXPROCS(0)
	if workers > rows {
		workers = rows
	}
	if workers < 1 {
		workers = 1
	}
	rowsPerWorker := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		rStart := w * rowsPerWorker
		rEnd := rStart + rowsPerWorker
		if rStart >= rows {
			break
		}
		if rEnd > rows {
			rEnd = rows
		}
		wg.Add(1)
		go func(rStart, rEnd int) {
			defer wg.Done()
			blockDeq := make([]float32, blockSize)
			for r := rStart; r < rEnd; r++ {
				rowOffset := r * rowBytes
				rowSlice := data[rowOffset : rowOffset+rowBytes]
				var sum float32
				for b := 0; b < blocksPerRow; b++ {
					bOffset := b * blockSizeBytes
					dequantFn(rowSlice[bOffset:bOffset+blockSizeBytes], blockDeq)
					vOffset := b * blockSize
					for k := 0; k < blockSize; k++ {
						sum += blockDeq[k] * vector[vOffset+k]
					}
				}
				result[r] = sum
			}
		}(rStart, rEnd)
	}
	wg.Wait()
	return result
}

// MatVecMulQ4_0 performs parallel zero-copy MatVec on Q4_0 data.
// Block layout: 18 bytes = 2 (f16 scale) + 16 (32 * 4-bit quants).
func MatVecMulQ4_0(data []byte, vector []float32, rows, cols int) []float32 {
	return matVecMulGeneric(data, vector, rows, cols, 32, 18, func(block []byte, dst []float32) {
		d := Float16ToFloat32(binary.LittleEndian.Uint16(block[0:2]))
		qs := block[2:18]
		for j := 0; j < 16; j++ {
			dst[j] = d * (float32(qs[j]&0x0F) - 8.0)
			dst[j+16] = d * (float32(qs[j]>>4) - 8.0)
		}
	})
}

// MatVecMulQ5_0 performs parallel zero-copy MatVec on Q5_0 data.
// Block layout: 22 bytes = 2 (f16) + 4 (uint32 high bits) + 16 (4-bit quants).
func MatVecMulQ5_0(data []byte, vector []float32, rows, cols int) []float32 {
	return matVecMulGeneric(data, vector, rows, cols, 32, 22, func(block []byte, dst []float32) {
		d := Float16ToFloat32(binary.LittleEndian.Uint16(block[0:2]))
		m := binary.LittleEndian.Uint32(block[2:6])
		qs := block[6:22]
		for j := 0; j < 16; j++ {
			high0 := (m >> uint(j)) & 1
			high1 := (m >> uint(j+16)) & 1
			dst[j] = d * (float32(uint8(qs[j]&0x0F)|uint8(high0<<4)) - 16.0)
			dst[j+16] = d * (float32(uint8(qs[j]>>4)|uint8(high1<<4)) - 16.0)
		}
	})
}

// MatVecMulQ2_K performs parallel zero-copy MatVec on Q2_K data.
// Block layout: 84 bytes = 16 (scales) + 64 (qs) + 2 (d f16) + 2 (dmin f16).
func MatVecMulQ2_K(data []byte, vector []float32, rows, cols int) []float32 {
	return matVecMulGeneric(data, vector, rows, cols, 256, 84, func(block []byte, dst []float32) {
		scales := block[0:16]
		qs := block[16:80]
		d := Float16ToFloat32(binary.LittleEndian.Uint16(block[80:82]))
		dmin := Float16ToFloat32(binary.LittleEndian.Uint16(block[82:84]))
		for l := 0; l < 16; l++ {
			sc := scales[l] & 0xF
			m := scales[l] >> 4
			dl := d * float32(sc)
			ml := dmin * float32(m)
			for k := 0; k < 16; k++ {
				idx := l*16 + k
				qByte := qs[idx/4]
				q := (qByte >> ((idx % 4) * 2)) & 3
				dst[idx] = dl*float32(q) - ml
			}
		}
	})
}

// MatVecMulQ3_K performs parallel zero-copy MatVec on Q3_K data.
// Block layout: 110 bytes = 32 (hmask) + 64 (qs 2-bit) + 12 (scales) + 2 (d f16).
func MatVecMulQ3_K(data []byte, vector []float32, rows, cols int) []float32 {
	return matVecMulGeneric(data, vector, rows, cols, 256, 110, func(block []byte, dst []float32) {
		hmask := block[0:32]
		qs := block[32:96]
		scales := block[96:108]
		d := Float16ToFloat32(binary.LittleEndian.Uint16(block[108:110]))

		var sc [16]uint8
		for j := 0; j < 4; j++ {
			sc[j] = scales[j] & 63
			sc[j+4] = scales[j+4] & 63
			sc[j+8] = scales[j+8] & 63
			sc[j+12] = (scales[j] >> 6) | ((scales[j+4] >> 6) << 2) | ((scales[j+8] >> 6) << 4)
		}
		for l := 0; l < 16; l++ {
			s := d * (float32(sc[l]) - 32.0)
			for k := 0; k < 16; k++ {
				idxInBlock := l*16 + k
				qsByte := qs[idxInBlock/4]
				q2 := (qsByte >> ((idxInBlock % 4) * 2)) & 3
				hmByte := hmask[idxInBlock/8]
				h := (hmByte >> (idxInBlock % 8)) & 1
				q := (h << 2) | q2
				dst[idxInBlock] = s * (float32(q) - 4.0)
			}
		}
	})
}

// MatVecMulQ5_K performs parallel zero-copy MatVec on Q5_K data.
// Block layout: 176 bytes = 2 (d f16) + 16 (scales) + 158 (qs).
func MatVecMulQ5_K(data []byte, vector []float32, rows, cols int) []float32 {
	return matVecMulGeneric(data, vector, rows, cols, 256, 176, func(block []byte, dst []float32) {
		d := Float16ToFloat32(binary.LittleEndian.Uint16(block[0:2]))
		scalesRaw := block[2:18]
		qs := block[18:176]

		var sc [8]uint8
		var m [8]uint8
		for j := 0; j < 8; j++ {
			sc[j] = scalesRaw[j] & 31
			m[j] = scalesRaw[j] >> 5
			if j < 4 {
				sc[j] |= (scalesRaw[j+8] & 1) << 5
				m[j] |= (scalesRaw[j+8] >> 1) << 3
			} else {
				sc[j] |= (scalesRaw[j+8] & 3) << 4
				m[j] |= (scalesRaw[j+8] >> 2) << 2
			}
		}
		for j := 0; j < 8; j++ {
			step := d * float32(sc[j]) / 31.0
			mj := float32(m[j])
			qsOffset := j * 20
			idxBase := j * 32
			for k := 0; k < 16 && qsOffset+k < len(qs); k++ {
				b := qs[qsOffset+k]
				if k < 8 {
					dst[idxBase+k] = step*float32(b&0x1F) - mj
					dst[idxBase+k+8] = step*float32((b>>5)&0x1F) - mj
				} else {
					idx := k - 8
					dst[idxBase+idx+8] = step*float32(b&0x1F) - mj
					if idx+16 < 32 {
						dst[idxBase+idx+16] = step*float32((b>>5)&0x1F) - mj
					}
				}
			}
		}
	})
}
