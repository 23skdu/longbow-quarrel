package device

import (
	"math"
	"testing"
)

func TestFloat16_Conversion(t *testing.T) {
	negZero := float32(math.Copysign(0, -1))
	testValues := []float32{0.0, negZero, 1.0, -1.0, 0.5, 65504.0, -65504.0, 0.0001}
	for _, v := range testValues {
		h := Float32ToFloat16(v)
		v2 := Float16ToFloat32(h)
		diff := math.Abs(float64(v - v2))
		if v != 0 && diff/math.Abs(float64(v)) > 0.05 {
			t.Errorf("Float16 conversion accuracy issue for %f: got %f", v, v2)
		}
	}

	// Test inf/nan
	inf := float32(math.Inf(1))
	hInf := Float32ToFloat16(inf)
	vInf := Float16ToFloat32(hInf)
	if !math.IsInf(float64(vInf), 1) && vInf < 65000 {
		t.Errorf("Expected Inf or clamped max, got %f", vInf)
	}

	// Test subnormal conversion branch
	sub := Float16ToFloat32(0x0001)
	if sub == 0 {
		t.Errorf("Expected non-zero subnormal conversion, got %f", sub)
	}
}

func TestFP8_E4M3_Conversion(t *testing.T) {
	values := []float32{0.0, 1.0, -1.0, 2.5, -3.75, 12.0, 440.0, 500.0}
	for _, v := range values {
		b := Float32ToFP8E4M3(v)
		v2 := FP8E4M3ToFloat32(b)
		if v == 0.0 && v2 != 0.0 {
			t.Errorf("expected 0.0, got %f", v2)
		}
		if v == 1.0 && math.Abs(float64(v2-1.0)) > 0.15 {
			t.Errorf("expected ~1.0, got %f", v2)
		}
	}

	// Subnormal and NaN
	subVal := FP8E4M3ToFloat32(0x01)
	if subVal == 0 {
		t.Errorf("expected non-zero for subnormal FP8 E4M3, got %f", subVal)
	}
	nanVal := FP8E4M3ToFloat32(0x7f)
	if !math.IsNaN(float64(nanVal)) {
		t.Errorf("expected NaN for 0x7f, got %f", nanVal)
	}
}

func TestFP8_E5M2_Conversion(t *testing.T) {
	values := []float32{0.0, 1.0, -2.0, 16.0, 1024.0, 65000.0}
	for _, v := range values {
		b := Float32ToFP8E5M2(v)
		v2 := FP8E5M2ToFloat32(b)
		if v == 0.0 && v2 != 0.0 {
			t.Errorf("expected 0.0, got %f", v2)
		}
		if v == 1.0 && math.Abs(float64(v2-1.0)) > 0.25 {
			t.Errorf("expected ~1.0, got %f", v2)
		}
	}

	// Subnormal and Inf
	subVal := FP8E5M2ToFloat32(0x01)
	if subVal == 0 {
		t.Errorf("expected non-zero for subnormal FP8 E5M2, got %f", subVal)
	}
	infVal := FP8E5M2ToFloat32(0x7c)
	if !math.IsInf(float64(infVal), 1) {
		t.Errorf("expected +Inf for 0x7c, got %f", infVal)
	}
}

func TestQ8_0_Quantization(t *testing.T) {
	src := []float32{1.5, -2.0, 0.0, 0.5, -1.0, 3.2, -4.5, 2.1}
	dst := make([]int8, len(src))
	scale := QuantizeBlockQ8_0(src, dst)
	if scale <= 0 {
		t.Fatalf("expected positive scale, got %f", scale)
	}

	dequant := make([]float32, len(src))
	DequantizeBlockQ8_0(dst, scale, dequant)

	for i := range src {
		diff := math.Abs(float64(src[i] - dequant[i]))
		if diff > 0.05 {
			t.Errorf("Q8_0 error at index %d: src=%f, dequant=%f", i, src[i], dequant[i])
		}
	}

	// Test zero vector
	zeros := make([]float32, 8)
	zeroScale := QuantizeBlockQ8_0(zeros, dst)
	if zeroScale != 0 {
		t.Errorf("expected 0 scale for zeros, got %f", zeroScale)
	}
}

func TestFloat32SliceToBytes(t *testing.T) {
	if b := Float32SliceToBytes(nil); b != nil {
		t.Errorf("expected nil for empty slice, got %v", b)
	}
	data := []float32{1.0, 2.0}
	b := Float32SliceToBytes(data)
	if len(b) != 8 {
		t.Errorf("expected 8 bytes, got %d", len(b))
	}
}
