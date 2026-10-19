package device

import (
	"encoding/binary"
	"fmt"
)

// DequantizeQ4K_Reference implements llama.cpp's Q4K dequantization in Go
func DequantizeQ4K_Reference(block []byte) []float32 {
	if len(block) != 144 {
		panic(fmt.Sprintf("Invalid Q4K block size: %d", len(block)))
	}

	out := make([]float32, 256)

	// Read d and dmin (FP16)
	d := Float16ToFloat32(binary.LittleEndian.Uint16(block[0:2]))
	dmin := Float16ToFloat32(binary.LittleEndian.Uint16(block[2:4]))

	// Read scales (12 bytes)
	scales := block[4:16]

	// Read quants (128 bytes)
	qs := block[16:144]

	// Decode scales (6-bit packed)
	// llama.cpp format: 8 scale values, each 6 bits, packed
	sc := make([]uint8, 8)
	m := make([]uint8, 8)

	// First 4 scales
	sc[0] = scales[0] & 0x3F
	sc[1] = scales[1] & 0x3F
	sc[2] = scales[2] & 0x3F
	sc[3] = scales[3] & 0x3F

	// First 4 mins
	m[0] = scales[4] & 0x3F
	m[1] = scales[5] & 0x3F
	m[2] = scales[6] & 0x3F
	m[3] = scales[7] & 0x3F

	// Next 4 scales (more complex packing)
	sc[4] = (scales[8] & 0x0F) | ((scales[0] & 0xC0) >> 2)
	sc[5] = (scales[9] & 0x0F) | ((scales[1] & 0xC0) >> 2)
	sc[6] = (scales[10] & 0x0F) | ((scales[2] & 0xC0) >> 2)
	sc[7] = (scales[11] & 0x0F) | ((scales[3] & 0xC0) >> 2)

	// Next 4 mins
	m[4] = (scales[8]&0xF0)>>4 | ((scales[4] & 0xC0) >> 2)
	m[5] = (scales[9]&0xF0)>>4 | ((scales[5] & 0xC0) >> 2)
	m[6] = (scales[10]&0xF0)>>4 | ((scales[6] & 0xC0) >> 2)
	m[7] = (scales[11]&0xF0)>>4 | ((scales[7] & 0xC0) >> 2)

	// Dequantize 256 values (8 groups of 32)
	for j := 0; j < 8; j++ {
		dVal := d * float32(sc[j])
		mVal := dmin * float32(m[j])

		// Each group has 32 values (16 bytes of packed 4-bit quants)
		for k := 0; k < 16; k++ {
			idx := j*16 + k
			q1 := qs[idx] & 0x0F
			q2 := (qs[idx] & 0xF0) >> 4

			out[j*32+k] = dVal*float32(q1) - mVal
			out[j*32+k+16] = dVal*float32(q2) - mVal
		}
	}

	return out
}
