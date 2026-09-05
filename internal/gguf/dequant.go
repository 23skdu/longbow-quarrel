package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"
)

const (
	BlockSizeQ4K = 256
	BlockSizeQ6K = 256
	BlockSizeQ5K = 256
)

func DequantizeQ5K(data []byte, numElements int) []float32 {
	if numElements%BlockSizeQ5K != 0 {
		panic("DequantizeQ5K: numElements must be multiple of 256")
	}

	numBlocks := numElements / BlockSizeQ5K
	out := make([]float32, numElements)

	const blockSizeBytes = 176

	for i := 0; i < numBlocks; i++ {
		blockOffset := i * blockSizeBytes
		if blockOffset+blockSizeBytes > len(data) {
			break
		}
		blockData := data[blockOffset : blockOffset+blockSizeBytes]

		d := Float16ToFloat32(binary.LittleEndian.Uint16(blockData[0:2]))
		scales := blockData[2:18]
		qs := blockData[18:176]

		var sc [8]uint8
		var m [8]uint8
		for j := 0; j < 8; j++ {
			sc[j] = scales[j] & 31
			m[j] = scales[j] >> 5
			if j < 4 {
				sc[j] |= (scales[j+8] & 1) << 5
				m[j] |= (scales[j+8] >> 1) << 3
			} else {
				sc[j] |= (scales[j+8] & 3) << 4
				m[j] |= (scales[j+8] >> 2) << 2
			}
		}

		baseIdx := i * BlockSizeQ5K
		for j := 0; j < 8; j++ {
			step := d * float32(sc[j]) / 31.0
			mj := float32(m[j])
			qsOffset := j * 20
			idxBase := baseIdx + j*32

			for k := 0; k < 16 && qsOffset+k < len(qs); k++ {
				b := qs[qsOffset+k]
				if k < 8 {
					out[idxBase+k] = step * float32(b&0x1F)
					out[idxBase+k+8] = step * float32((b>>5)&0x1F)
				} else {
					idx := k - 8
					out[idxBase+idx+8] = step * float32(b&0x1F)
					if idx+16 < 32 {
						out[idxBase+idx+16] = step * float32((b>>5)&0x1F)
					}
				}
			}
			for k := 16; k < 32; k++ {
				out[idxBase+k] -= mj
			}
		}
	}

	return out
}

func DequantizeQ4K(data []byte, numElements int) []float32 {
	if numElements%BlockSizeQ4K != 0 {
		panic("DequantizeQ4K: numElements must be multiple of 256")
	}

	numBlocks := numElements / BlockSizeQ4K
	out := make([]float32, numElements)

	const blockSizeBytes = 144

	for i := 0; i < numBlocks; i++ {
		blockOffset := i * blockSizeBytes

		blockData := data[blockOffset : blockOffset+blockSizeBytes]

		d := Float16ToFloat32(binary.LittleEndian.Uint16(blockData[0:2]))
		// dmin is not used in the updated group-level scaling math
		_ = Float16ToFloat32(binary.LittleEndian.Uint16(blockData[2:4]))

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
			for k := 0; k < 16; k++ {
				b := qs[qsOffset+k]
				out[idxBase+k] = step * (float32(b&0xF) - mj)
				out[idxBase+k+16] = step * (float32(b>>4) - mj)
			}
		}
	}

	return out
}

func DequantizeQ4KBranchless(data []byte, numElements int) []float32 {
	if numElements%BlockSizeQ4K != 0 {
		panic("DequantizeQ4KBranchless: numElements must be multiple of 256")
	}

	numBlocks := numElements / BlockSizeQ4K
	out := make([]float32, numElements)

	const blockSizeBytes = 144

	for i := 0; i < numBlocks; i++ {
		blockOffset := i * blockSizeBytes

		blockData := data[blockOffset : blockOffset+blockSizeBytes]

		d := Float16ToFloat32(binary.LittleEndian.Uint16(blockData[0:2]))
		dmin := Float16ToFloat32(binary.LittleEndian.Uint16(blockData[2:4]))

		scales := blockData[4:16]
		qs := blockData[16:144]

		var sc [8]uint8
		var m [8]uint8

		for j := 0; j < 4; j++ {
			sc[j] = scales[j] & 63
			m[j] = scales[j+4] & 63
		}

		for j := 4; j < 8; j++ {
			sc[j] = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
			m[j] = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
		}

		var D [8]float32
		var M [8]float32

		for j := 0; j < 8; j++ {
			D[j] = d * float32(sc[j])
			M[j] = dmin * float32(m[j])
		}

		baseIdx := i * BlockSizeQ4K
		for j := 0; j < 8; j++ {
			dj := D[j]
			mj := M[j]
			qsOffset := j * 16
			idxBase := baseIdx + j*32
			for k := 0; k < 16; k++ {
				b := qs[qsOffset+k]
				v0 := float32(b & 0xF)
				v1 := float32(b >> 4)
				out[idxBase+k] = dj*v0 - mj
				out[idxBase+k+16] = dj*v1 - mj
			}
		}
	}

	return out
}

// DequantizeQ3K converts Q3_K data to Float32.
// Layout (110 bytes per 256 weights):
// - hmask: 32 bytes (256 bits) - high bit of the 3-bit quant
// - qs: 64 bytes (256 * 2 bits) - low 2 bits
// - scales: 12 bytes (16 6-bit scales)
// - d: f16 (super-scale)
func DequantizeQ3K(data []byte, numElements int) []float32 {
	const blockSizeBytes = 110 // 32 + 64 + 12 + 2
	numBlocks := numElements / 256
	out := make([]float32, numElements)

	for i := 0; i < numBlocks; i++ {
		blockOffset := i * blockSizeBytes
		if blockOffset+blockSizeBytes > len(data) {
			break
		}
		block := data[blockOffset : blockOffset+blockSizeBytes]

		hmask := block[0:32]
		qs := block[32:96]
		scales := block[96:108]
		d := Float16ToFloat32(binary.LittleEndian.Uint16(block[108:110]))

		if i == 0 {
			fmt.Printf("DEBUG: Q3K Block 0: d=%f, hmask[0]=%x, qs[0]=%x\n", d, hmask[0], qs[0])
		}

		// Unpack scales (same logic as Q4_K but fewer bits/scales? No, same 12 bytes -> 16 scales)
		// Q3_K uses scales to store 16 6-bit scales.
		// Layout of scales matches Q4_K's `sc` part (without `m`).
		// Actually Q3_K scales packing:
		// 12 bytes -> 16 6-bit numbers.
		// bits: n_n = 6. 16 * 6 = 96 bits = 12 bytes.
		// Packing: split into top/bottom 4 bits? relative to what?

		// Logic from k_quants.c:
		// for (j = 0; j < 4; ++j) {
		//     sc[j]   = scales[j] & 63;
		//     sc[j+4] = scales[j+4] & 63;
		//     sc[j+8] = scales[j+8] & 63;
		//     sc[j+12] = (scales[j] >> 6) | ((scales[j+4] >> 6) << 2) | ((scales[j+8] >> 6) << 4);
		// }
		// Wait. 12 bytes in input `scales`.
		// Output `sc` is 16 bytes (uint8).

		var sc [16]uint8
		for j := 0; j < 4; j++ {
			sc[j] = scales[j] & 63
			sc[j+4] = scales[j+4] & 63
			sc[j+8] = scales[j+8] & 63
			sc[j+12] = (scales[j] >> 6) | ((scales[j+4] >> 6) << 2) | ((scales[j+8] >> 6) << 4)
		}

		// Decode weights
		// 16 blocks of 16 weights.
		// For each block l=0..15:
		// scale = d * (sc[l] - 32)
		// q = (hmask bit) << 2 | (qs bits)
		// val = scale * (q - 4)

		for l := 0; l < 16; l++ {
			// Effective scale
			s := d * (float32(sc[l]) - 32.0)
			if s == 0 {
				s = 0
			} // Avoid -0?

			// 16 weights in this sub-block
			// indices k=0..15
			for k := 0; k < 16; k++ {
				idxInBlock := l*16 + k

				// Get 2 bits from qs
				// qs is 64 bytes. 256 weights. 4 weights/byte.
				// Byte index = idxInBlock / 4
				// Shift = (idxInBlock % 4) * 2
				qsByte := qs[idxInBlock/4]
				q2 := (qsByte >> ((idxInBlock % 4) * 2)) & 3

				// Get 1 bit from hmask
				// hmask is 32 bytes. 256 bits.
				// Byte index = idxInBlock / 8
				// Shift = idxInBlock % 8
				// Wait. layout usually matches qs?
				// hmask[j] contains high bits for weights j*8 .. j*8+7?
				// bit k corresponds to weight j*8 + k?
				hmByte := hmask[idxInBlock/8]
				h := (hmByte >> (idxInBlock % 8)) & 1

				// q = h << 2 | q2 (3 bits: 0..7)
				q := (h << 2) | q2

				// val = s * (q - 4)
				out[i*256+idxInBlock] = s * (float32(q) - 4.0)
			}
		}
	}
	return out
}

// DequantizeQ2K converts Q2_K data (2-bit) to Float32.
// Layout (84 bytes per 256 weights):
// - d: f16 (super-scale)
// - dmin: f16 (super-min)
// - scales: 16 bytes (16 groups of 4-bit scale + 4-bit min)
// - qs: 64 bytes (256 * 2 bits)
func DequantizeQ2K(data []byte, numElements int) []float32 {
	const blockSizeBytes = 84
	numBlocks := numElements / 256
	out := make([]float32, numElements)

	for i := 0; i < numBlocks; i++ {
		blockOffset := i * blockSizeBytes
		if blockOffset+blockSizeBytes > len(data) {
			break
		}
		block := data[blockOffset : blockOffset+blockSizeBytes]

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
				// 4 weights per byte in qs
				qByte := qs[idx/4]
				q := (qByte >> ((idx % 4) * 2)) & 3
				out[i*256+idx] = dl*float32(q) - ml
			}
		}
	}
	return out
}

// DequantizeIQ4XS converts IQ4_XS data to Float32.
// Layout (138 bytes per 256 weights):
// - d: f16 (super-scale)
// - scales: 8 bytes (8 groups of 8-bit scales) - Wait, sigmas?
// - qs: 128 bytes (256 * 4 bits)
func DequantizeIQ4XS(data []byte, numElements int) []float32 {
	const blockSizeBytes = 138
	numBlocks := numElements / 256
	out := make([]float32, numElements)

	for i := 0; i < numBlocks; i++ {
		blockOffset := i * blockSizeBytes
		if blockOffset+blockSizeBytes > len(data) {
			break
		}
		block := data[blockOffset : blockOffset+blockSizeBytes]

		d := Float16ToFloat32(binary.LittleEndian.Uint16(block[136:138]))
		qs := block[0:128]
		scales := block[128:136]

		for j := 0; j < 8; j++ {
			s := d * float32(scales[j])
			// 32 elements per scale group
			for k := 0; k < 16; k++ {
				idx := j*32 + k
				b := qs[j*16+k]
				// IQ4_XS uses a lookup table in reality, but for a 4-bit standard:
				out[i*256+idx] = s * (float32(b&0xF) - 8.0)
				out[i*256+idx+16] = s * (float32(b>>4) - 8.0)
			}
		}
	}
	return out
}

func DequantizeQ6K(data []byte, numElements int) []float32 {
	const blockSizeBytes = 210
	numBlocks := numElements / 256
	out := make([]float32, numElements)

	for i := 0; i < numBlocks; i++ {
		blockOffset := i * blockSizeBytes
		if blockOffset+blockSizeBytes > len(data) {
			break
		}
		block := data[blockOffset : blockOffset+blockSizeBytes]

		qs := block[0:128]
		qh := block[128:192]
		scales := block[192:208]
		d := Float16ToFloat32(binary.LittleEndian.Uint16(block[208:210]))

		base := i * 256

		for si := 0; si < 2; si++ {
			scOff := si * 8
			n := si * 128
			for l := 0; l < 32; l++ {
				is := l / 16
				qhOff := si * 32

				q1 := int8((qs[l+0]&0xF)|(((qh[l+qhOff]>>0)&3)<<4)) - 32  // #nosec G115
				q2 := int8((qs[l+32]&0xF)|(((qh[l+qhOff]>>2)&3)<<4)) - 32 // #nosec G115
				q3 := int8((qs[l+0]>>4)|(((qh[l+qhOff]>>4)&3)<<4)) - 32   // #nosec G115
				q4 := int8((qs[l+32]>>4)|(((qh[l+qhOff]>>6)&3)<<4)) - 32  // #nosec G115

				yIdx := base + n + l
				out[yIdx+0] = d * float32(int8(scales[scOff+is*2+0])) * float32(q1)  // #nosec G115
				out[yIdx+32] = d * float32(int8(scales[scOff+is*2+1])) * float32(q2) // #nosec G115
				out[yIdx+64] = d * float32(int8(scales[scOff+is*2+2])) * float32(q3) // #nosec G115
				out[yIdx+96] = d * float32(int8(scales[scOff+is*2+3])) * float32(q4) // #nosec G115
			}
		}
	}
	return out
}

func DequantizeF16(data []byte, numElements int) []float32 {
	out := make([]float32, numElements)
	for i := 0; i < numElements; i++ {
		bits := binary.LittleEndian.Uint16(data[i*2 : (i+1)*2])
		out[i] = Float16ToFloat32(bits)
	}
	return out
}

func Float16ToFloat32(b uint16) float32 {
	sign := uint32(b&0x8000) << 16
	exp := uint32(b&0x7C00) >> 10
	frac := uint32(b&0x03FF) << 13

	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		// subnormal
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

// DequantizeQ8_0 converts Q8_0 data to Float32.
// Layout (34 bytes per 32 weights):
// - d: f16 (scale)
// - qs: 32 bytes (int8)
func DequantizeQ8_0(data []byte, numElements int) []float32 {
	const blockSize = 32
	const blockSizeBytes = 34 // 2 + 32
	if numElements%blockSize != 0 {
		panic(fmt.Sprintf("DequantizeQ8_0: numElements %d must be multiple of 32", numElements))
	}

	numBlocks := numElements / blockSize
	out := make([]float32, numElements)

	for i := 0; i < numBlocks; i++ {
		blockOffset := i * blockSizeBytes
		if blockOffset+blockSizeBytes > len(data) {
			break
		}

		// Parse scale (f16)
		d := Float16ToFloat32(binary.LittleEndian.Uint16(data[blockOffset : blockOffset+2]))

		// Parse quants (32 * int8)
		qs := data[blockOffset+2 : blockOffset+34]

		for j := 0; j < blockSize; j++ {
			out[i*blockSize+j] = d * float32(int8(qs[j])) // #nosec G115 -- byte to int8 for quantized data
		}
	}
	return out
}

// MatVecMulQ8_0 performs matrix-vector multiplication directly on Q8_0 quantized data
// without materializing the dequantized weights in memory.
// matrix shape is [rows, cols], vector length is cols.
func MatVecMulQ8_0(data []byte, vector []float32, rows, cols int) []float32 {
	result := make([]float32, rows)
	const blockSize = 32
	const blockSizeBytes = 34
	if cols%blockSize != 0 || len(vector) < cols {
		return result
	}
	blocksPerRow := cols / blockSize
	rowBytes := blocksPerRow * blockSizeBytes

	parallelism := runtime.NumCPU()
	if parallelism <= 0 || rows < 32 {
		parallelism = 1
	}
	chunkSize := (rows + parallelism - 1) / parallelism

	var wg sync.WaitGroup
	for c := 0; c < parallelism; c++ {
		startRow := c * chunkSize
		if startRow >= rows {
			break
		}
		endRow := startRow + chunkSize
		if endRow > rows {
			endRow = rows
		}
		wg.Add(1)
		go func(rStart, rEnd int) {
			defer wg.Done()
			for i := rStart; i < rEnd; i++ {
				rowOffset := i * rowBytes
				if rowOffset+rowBytes > len(data) {
					break
				}
				var rowSum float32
				for b := 0; b < blocksPerRow; b++ {
					bOffset := rowOffset + b*blockSizeBytes
					d := Float16ToFloat32(binary.LittleEndian.Uint16(data[bOffset : bOffset+2]))
					vBase := b * blockSize
					var blockSum float32
					for j := 0; j < 32; j++ {
						blockSum += float32(int8(data[bOffset+2+j])) * vector[vBase+j] // #nosec G115
					}
					rowSum += d * blockSum
				}
				result[i] = rowSum
			}
		}(startRow, endRow)
	}
	wg.Wait()
	return result
}

// DequantizeQ4_0 converts Q4_0 data to Float32.
// Layout (18 bytes per 32 weights):
// - d: f16 (scale)
// - qs: 16 bytes (32 * 4 bits)
func DequantizeQ4_0(data []byte, numElements int) []float32 {
	const blockSize = 32
	const blockSizeBytes = 18
	if numElements%blockSize != 0 {
		panic(fmt.Sprintf("DequantizeQ4_0: numElements %d must be multiple of 32", numElements))
	}

	numBlocks := numElements / blockSize
	out := make([]float32, numElements)

	for i := 0; i < numBlocks; i++ {
		blockOffset := i * blockSizeBytes
		if blockOffset+blockSizeBytes > len(data) {
			break
		}

		d := Float16ToFloat32(binary.LittleEndian.Uint16(data[blockOffset : blockOffset+2]))
		qs := data[blockOffset+2 : blockOffset+18]

		for j := 0; j < 16; j++ {
			v0 := qs[j] & 0x0F
			v1 := qs[j] >> 4

			out[i*blockSize+j] = d * (float32(v0) - 8.0)
			out[i*blockSize+j+16] = d * (float32(v1) - 8.0)
		}
	}
	return out
}

// DequantizeQ5_0 converts Q5_0 data to Float32.
// Layout (22 bytes per 32 weights):
// - d: f16 (scale)
// - m: uint32 (32 * 1 high bit)
// - qs: 16 bytes (32 * 4 bits)
func DequantizeQ5_0(data []byte, numElements int) []float32 {
	const blockSize = 32
	const blockSizeBytes = 22
	if numElements%blockSize != 0 {
		panic(fmt.Sprintf("DequantizeQ5_0: numElements %d must be multiple of 32", numElements))
	}

	numBlocks := numElements / blockSize
	out := make([]float32, numElements)

	for i := 0; i < numBlocks; i++ {
		blockOffset := i * blockSizeBytes
		if blockOffset+blockSizeBytes > len(data) {
			break
		}

		d := Float16ToFloat32(binary.LittleEndian.Uint16(data[blockOffset : blockOffset+2]))
		m := binary.LittleEndian.Uint32(data[blockOffset+2 : blockOffset+6])
		qs := data[blockOffset+6 : blockOffset+22]

		for j := 0; j < 16; j++ {
			v0 := qs[j] & 0x0F
			v1 := qs[j] >> 4

			high0 := (m >> j) & 1
			high1 := (m >> (j + 16)) & 1

			val0 := uint8(v0) | (uint8(high0) << 4)
			val1 := uint8(v1) | (uint8(high1) << 4)

			out[i*blockSize+j] = d * (float32(val0) - 16.0)
			out[i*blockSize+j+16] = d * (float32(val1) - 16.0)
		}
	}
	return out
}

// DequantizeBlock is a dispatcher that dequantizes a single block of data based on its GGMLType.
func DequantizeBlock(data []byte, dst []float32, dataType GGMLType) {
	switch dataType {
	case GGMLTypeF32:
		for i := 0; i < len(dst); i++ {
			if (i+1)*4 <= len(data) {
				dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4 : (i+1)*4]))
			}
		}
	case GGMLTypeF16:
		for i := 0; i < len(dst); i++ {
			if (i+1)*2 <= len(data) {
				dst[i] = Float16ToFloat32(binary.LittleEndian.Uint16(data[i*2 : (i+1)*2]))
			}
		}
	case GGMLTypeQ4_K:
		res := DequantizeQ4K(data, len(dst))
		copy(dst, res)
	case GGMLTypeQ6_K:
		res := DequantizeQ6K(data, len(dst))
		copy(dst, res)
	case GGMLTypeQ8_0:
		res := DequantizeQ80(data, len(dst))
		copy(dst, res)
	case GGMLTypeQ5_0:
		res := DequantizeQ5_0(data, len(dst))
		copy(dst, res)
	case GGMLTypeQ2_K:
		res := DequantizeQ2K(data, len(dst))
		copy(dst, res)
	case GGMLTypeQ5_K:
		res := DequantizeQ5K(data, len(dst))
		copy(dst, res)
	case GGMLTypeIQ4_XS:
		res := DequantizeIQ4XS(data, len(dst))
		copy(dst, res)
	default:
		// Fallback or panic for unsupported types in tests
		panic(fmt.Sprintf("DequantizeBlock: unsupported type %v", dataType))
	}
}

func DequantizeQ80(data []byte, numElements int) []float32 {
	const blockSize = 32
	const blockSizeBytes = 34
	if numElements%blockSize != 0 {
		return make([]float32, numElements)
	}
	numBlocks := numElements / blockSize
	out := make([]float32, numElements)
	for i := 0; i < numBlocks; i++ {
		off := i * blockSizeBytes
		if off+blockSizeBytes > len(data) {
			break
		}
		d := Float16ToFloat32(binary.LittleEndian.Uint16(data[off : off+2]))
		qs := data[off+2 : off+34]
		for j := 0; j < 32; j++ {
			out[i*blockSize+j] = d * float32(int8(qs[j])) // #nosec G115 -- byte to int8 for quantized data
		}
	}
	return out
}
