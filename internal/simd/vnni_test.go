package simd

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// float32ToFloat16 converts float32 to float16 binary format for testing
func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xFF) - 127 + 15
	frac := uint16((bits >> 13) & 0x03FF)

	if exp <= 0 {
		return sign
	}
	if exp >= 31 {
		return sign | 0x7C00
	}
	return sign | uint16(exp<<10) | frac
}

func TestSIMDCPUInfoFlags(t *testing.T) {
	// Verify that detecting CPU info does not panic and returns boolean values
	_ = HasAMX()
	_ = HasAVXVNNI()
	level := GetCPULevel()
	if level < CPULevelScalar || level > CPULevelAVX512 {
		t.Errorf("Unexpected CPU level: %d", level)
	}
}

func TestVecDotQ8_0_VNNI(t *testing.T) {
	const numBlocks = 4
	const cols = numBlocks * 32
	const rowBytes = numBlocks * 34

	data := make([]byte, rowBytes)
	vec := make([]float32, cols)
	rng := rand.New(rand.NewSource(42))

	var expectedDot float32
	for b := 0; b < numBlocks; b++ {
		scale := float32(0.5 + rng.Float32()*1.5)
		binary.LittleEndian.PutUint16(data[b*34:b*34+2], float32ToFloat16(scale))

		for j := 0; j < 32; j++ {
			qVal := int8(rng.Intn(255) - 128)
			data[b*34+2+j] = byte(qVal)

			vVal := (rng.Float32() - 0.5) * 2.0
			vec[b*32+j] = vVal

			expectedDot += (scale * float32(qVal)) * vVal
		}
	}

	gotDot := VecDotQ8_0_VNNI(data, vec)
	diff := math.Abs(float64(gotDot - expectedDot))
	relErr := diff / (math.Abs(float64(expectedDot)) + 1e-6)
	if relErr > 0.05 && diff > 1e-2 {
		t.Errorf("VecDotQ8_0_VNNI got %f, want %f (relErr %f)", gotDot, expectedDot, relErr)
	}

	// Test MatVecMulQ8_0_VNNI
	rows := 2
	matrixData := make([]byte, rows*rowBytes)
	copy(matrixData[:rowBytes], data)
	copy(matrixData[rowBytes:], data)

	matVecRes := MatVecMulQ8_0_VNNI(matrixData, vec, rows, cols)
	if len(matVecRes) != rows {
		t.Fatalf("MatVecMulQ8_0_VNNI expected %d results, got %d", rows, len(matVecRes))
	}
	if math.Abs(float64(matVecRes[0]-gotDot)) > 1e-5 {
		t.Errorf("MatVecMulQ8_0_VNNI row 0 mismatch: got %f, want %f", matVecRes[0], gotDot)
	}
}

func TestVecDotQ4_K_VNNI(t *testing.T) {
	const numBlocks = 2
	const cols = numBlocks * 256
	const rowBytes = numBlocks * 144

	data := make([]byte, rowBytes)
	vec := make([]float32, cols)
	rng := rand.New(rand.NewSource(123))

	for i := 0; i < cols; i++ {
		vec[i] = (rng.Float32() - 0.5) * 1.5
	}

	for b := 0; b < numBlocks; b++ {
		bOffset := b * 144
		d := float32(1.2)
		binary.LittleEndian.PutUint16(data[bOffset:bOffset+2], float32ToFloat16(d))

		for j := 0; j < 12; j++ {
			data[bOffset+4+j] = byte(rng.Intn(256))
		}
		for j := 0; j < 128; j++ {
			data[bOffset+16+j] = byte(rng.Intn(256))
		}
	}

	gotDot := VecDotQ4_K_VNNI(data, vec)
	if math.IsNaN(float64(gotDot)) || math.IsInf(float64(gotDot), 0) {
		t.Errorf("VecDotQ4_K_VNNI produced NaN or Inf: %f", gotDot)
	}

	// MatVecMulQ4_K_VNNI
	rows := 2
	matrixData := make([]byte, rows*rowBytes)
	copy(matrixData[:rowBytes], data)
	copy(matrixData[rowBytes:], data)

	matVecRes := MatVecMulQ4_K_VNNI(matrixData, vec, rows, cols)
	if len(matVecRes) != rows {
		t.Fatalf("MatVecMulQ4_K_VNNI expected %d rows, got %d", rows, len(matVecRes))
	}
	if math.Abs(float64(matVecRes[0]-gotDot)) > 1e-5 {
		t.Errorf("MatVecMulQ4_K_VNNI row 0 mismatch: got %f, want %f", matVecRes[0], gotDot)
	}
}
