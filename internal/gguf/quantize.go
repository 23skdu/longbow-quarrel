package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
)

const (
	Q4KBlockSize      = 256
	Q4KBlockSizeBytes = 144
)

func Float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xFF) - 127 + 15
	frac := bits & 0x7FFFFF

	if exp <= 0 {
		exp = 0
		frac = 0
	} else if exp >= 31 {
		exp = 31
		frac = 0
	}

	return sign | uint16(exp<<10) | uint16(frac>>13)
}

func QuantizeWeightsToQ4K(weights []float32, numElements int) ([]byte, error) {
	if numElements%Q4KBlockSize != 0 {
		return nil, fmt.Errorf("QuantizeWeightsToQ4K: numElements %d must be multiple of %d", numElements, Q4KBlockSize)
	}

	numBlocks := numElements / Q4KBlockSize
	out := make([]byte, numBlocks*Q4KBlockSizeBytes)

	for i := 0; i < numBlocks; i++ {
		blockStart := i * Q4KBlockSize
		blockWeights := weights[blockStart : blockStart+Q4KBlockSize]

		var d float32 = 0
		for _, w := range blockWeights {
			absW := float32(math.Abs(float64(w)))
			if absW > d {
				d = absW
			}
		}

		if d == 0 {
			d = 1.0
		}

		dmin := d
		for _, w := range blockWeights {
			if w < -dmin {
				dmin = -w
			}
		}
		if dmin == 0 {
			dmin = d
		}

		blockOffset := i * Q4KBlockSizeBytes
		binary.LittleEndian.PutUint16(out[blockOffset:blockOffset+2], Float32ToFloat16(d))
		binary.LittleEndian.PutUint16(out[blockOffset+2:blockOffset+4], Float32ToFloat16(dmin))

		scales := out[blockOffset+4 : blockOffset+16]
		qs := out[blockOffset+16 : blockOffset+Q4KBlockSizeBytes]

		var sc [16]uint8
		var m [16]uint8

		for g := 0; g < 8; g++ {
			groupStart := g * 32

			var groupMinVal float32 = 0
			var groupMaxVal float32 = 0
			for _, w := range blockWeights[groupStart : groupStart+32] {
				if w < groupMinVal {
					groupMinVal = w
				}
				if w > groupMaxVal {
					groupMaxVal = w
				}
			}

			rangeVal := groupMaxVal - groupMinVal
			if rangeVal == 0 {
				rangeVal = d
			}

			sc[g] = uint8(math.Round(float64(rangeVal / d * 15.0)))
			if sc[g] == 0 {
				sc[g] = 1
			}

			m[g] = uint8(math.Round(float64(-groupMinVal / rangeVal * 15.0)))
		}

		for j := 0; j < 8; j++ {
			sc[j] = sc[j] & 63
		}

		for j := 8; j < 12; j++ {
			sc[j-4] = (sc[j-4] & 0x0F) | ((sc[j-8] >> 6) << 4)
		}

		for j := 0; j < 8; j++ {
			m[j] = m[j] & 63
		}

		for j := 8; j < 12; j++ {
			m[j-4] = (m[j-4] & 0x0F) | ((m[j-8] >> 6) << 4)
		}

		scales[0] = sc[0]
		scales[1] = sc[1]
		scales[2] = sc[2]
		scales[3] = sc[3]
		scales[4] = m[0]
		scales[5] = m[1]
		scales[6] = m[2]
		scales[7] = m[3]
		scales[8] = (sc[4] & 0x0F) | ((m[4] & 0x0F) << 4)
		scales[9] = (sc[5] & 0x0F) | ((m[5] & 0x0F) << 4)
		scales[10] = (sc[6] & 0x0F) | ((m[6] & 0x0F) << 4)
		scales[11] = (sc[7] & 0x0F) | ((m[7] & 0x0F) << 4)

		for g := 0; g < 8; g++ {
			groupStart := g * 32

			var groupMinVal float32 = 0
			for _, w := range blockWeights[groupStart : groupStart+32] {
				if w < groupMinVal {
					groupMinVal = w
				}
			}

			D := d * float32(sc[g]) / 15.0
			offset := -groupMinVal

			for j := 0; j < 32; j++ {
				w := blockWeights[groupStart+j]

				q := int8(math.Round(float64((w + offset) / D)))
				if q > 15 {
					q = 15
				}
				if q < 0 {
					q = 0
				}

				qsIdx := g*16 + j
				if j < 16 {
					qs[qsIdx] = uint8(q) & 0x0F
				} else {
					qs[qsIdx-16] |= (uint8(q) & 0x0F) << 4
				}
			}
		}
	}

	return out, nil
}

func DequantizeWeightsFromQ4K(data []byte, rows, cols int) ([]float32, error) {
	numElements := rows * cols
	return DequantizeQ4K(data, numElements), nil
}
