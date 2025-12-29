package gguf
import (
	"encoding/binary"
	"math"
	"fmt"
)

// Block sizes
const (
	BlockSizeQ4K = 256
	BlockSizeQ6K = 256
)

// DequantizeQ4K converts a Q4_K quantized tensor data to Float32.
// Layout:
// - d (f16): super-block scale
// - dmin (f16): super-block min
// - scales (12 bytes): 3-bit K-scales (mixed with L-scales)
// - qs (128 bytes): 4-bit quants
func DequantizeQ4K(data []byte, numElements int) []float32 {
	if numElements%BlockSizeQ4K != 0 {
		panic("DequantizeQ4K: numElements must be multiple of 256")
	}
	
	numBlocks := numElements / BlockSizeQ4K
	out := make([]float32, numElements)
	
	// Size of one Q4_K block is 144 bytes
	const blockSizeBytes = 144
	
	for i := 0; i < numBlocks; i++ {
		blockOffset := i * blockSizeBytes
		if blockOffset+blockSizeBytes > len(data) {
			break
		}
		
		blockData := data[blockOffset : blockOffset+blockSizeBytes]
		
		// Parse header
		d := float16ToFloat32(binary.LittleEndian.Uint16(blockData[0:2]))
		dmin := float16ToFloat32(binary.LittleEndian.Uint16(blockData[2:4]))
		
		scales := blockData[4:16] // 12 bytes
		qs := blockData[16:144]   // 128 bytes
		
		// Reference unpacking logic for Q4_K
		// scales is 12 bytes. We extract 8 scales (sc) and 8 mins (m).
		var sc [8]uint8
		var m  [8]uint8
		
		// Extract scales and mins using llama.cpp's get_scale_min_k4 logic
		for j := 0; j < 8; j++ {
			if j < 4 {
				sc[j] = scales[j] & 63
				m[j] = scales[j+4] & 63
			} else {
				sc[j] = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
				m[j] = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
			}
		}
		
		// Precompute effective scales/mins
		var D [8]float32
		var M [8]float32
		
		for j := 0; j < 8; j++ {
			D[j] = d * float32(sc[j])
			M[j] = dmin * float32(m[j])
		}
		
		// Decode quants
		for j := 0; j < 8; j++ {
			// For each sub-block j (32 weights)
			qsOffset := j * 16 // 16 bytes
			for k := 0; k < 16; k++ {
				// Byte k contains low nibble for weight k and high nibble for weight k+16
				b := qs[qsOffset+k]
				v0 := b & 0xF
				v1 := b >> 4
				
				// Weight indices relative to sub-block start
				idx0 := j*32 + k
				idx1 := idx0 + 16
				
				out[i*BlockSizeQ4K + idx0] = D[j] * float32(v0) - M[j]
				out[i*BlockSizeQ4K + idx1] = D[j] * float32(v1) - M[j]
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
		if blockOffset+blockSizeBytes > len(data) { break }
		block := data[blockOffset : blockOffset+blockSizeBytes]
		
		hmask := block[0:32]
		qs := block[32:96]
		scales := block[96:108]
		d := float16ToFloat32(binary.LittleEndian.Uint16(block[108:110]))
		
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
			sc[j]    = scales[j] & 63
			sc[j+4]  = scales[j+4] & 63
			sc[j+8]  = scales[j+8] & 63
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
			if s == 0 { s = 0 } // Avoid -0?
			
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
				out[i*256 + idxInBlock] = s * (float32(q) - 4.0)
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
		if blockOffset+blockSizeBytes > len(data) { break }
		block := data[blockOffset : blockOffset+blockSizeBytes]
		
		// Q6_K layout (210 bytes):
		// - qs: 128 bytes (256 * 4 bits) - low 4 bits
		// - qh: 64 bytes (256 * 2 bits) - high 2 bits
		// - scales: 16 bytes (16 8-bit scales)
		// - d: f16 (super-scale)
		// Wait. standard k-quants Q6_K:
		// ql (128)
		// qh (64)
		// scales (16) (int8)
		// d (f16)
		// Total 128+64+16+2 = 210.
		
		// Logic from k_quants.c:
		// const uint8_t * ql = (const uint8_t *) x->ql;
		// const uint8_t * qh = (const uint8_t *) x->qh;
		// const int8_t  * sc = (const int8_t  *) x->scales;
		// float d = GGML_FP16_TO_FP32(x->d);
		
		qs := block[0:128]
		qh := block[128:192]
		scales := block[192:208]
		d := float16ToFloat32(binary.LittleEndian.Uint16(block[208:210]))
		
		for l := 0; l < 16; l++ {
			// scale = d * sc[l]
			s := d * float32(int8(scales[l]))
			
			for k := 0; k < 16; k++ {
				idx := l*16 + k
				
				// low 4 bits from qs
				// qs[j] has 2 weights (low nibble, high nibble)? 
				// No, qs is 128 bytes for 256 weights. 2 weights per byte.
				// weight j: if j%2==0 low 4 bits, else high 4 bits?
				// standard packing:
				// ql[j]: weights 2*j and 2*j+1.
				// w[2*j] = ql[j] & 0xF
				// w[2*j+1] = ql[j] >> 4
				
				byteIdx := idx / 2
				qsByte := qs[byteIdx]
				var q4 uint8
				if idx % 2 == 0 {
					q4 = qsByte & 0x0F
				} else {
					q4 = (qsByte & 0xF0) >> 4
				}
				
				// high 2 bits from qh
				// qh is 64 bytes for 256 weights. 4 weights per byte.
				// weights j, j+1, j+2, j+3 encoded in qh[j/4]?
				// w[4*j]   : bits 0,1
				// w[4*j+1] : bits 2,3
				// ...
				qhByte := qh[idx / 4]
				shift := (idx % 4) * 2
				q2 := (qhByte >> shift) & 0x03
				
				// Combined 6 bits
				q := int8((q2 << 4) | q4)
				
				// val = s * (q - 32)
				out[i*256 + idx] = s * (float32(q) - 32.0)
			}
		}
	}
	return out
}

func float16ToFloat32(b uint16) float32 {
	sign := uint32(b & 0x8000) << 16
	exp := uint32(b & 0x7C00) >> 10
	frac := uint32(b & 0x03FF) << 13
	
	if exp == 0 {
		if frac == 0 { return math.Float32frombits(sign) }
		// subnormal
		f := float64(frac) * math.Pow(2, -24)
		if sign != 0 { f = -f }
		return float32(f * math.Pow(2, -14))
	} else if exp == 0x1F {
		if frac == 0 {
			if sign != 0 { return float32(math.Inf(-1)) }
			return float32(math.Inf(1))
		}
		return float32(math.NaN())
	}
	
	return math.Float32frombits(sign | ((exp + 112) << 23) | frac)
}
