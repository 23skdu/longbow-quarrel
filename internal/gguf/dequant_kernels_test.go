package gguf

import (
	"encoding/binary"
	"testing"
)

func TestDequantizeF16_Kernel(t *testing.T) {
	// 2 elements. F16 is 2 bytes per element.
	data := []byte{
		0x00, 0x3c, // 1.0 in F16
		0x00, 0xc0, // -2.0 in F16
	}
	res := DequantizeF16(data, 2)
	if len(res) != 2 {
		t.Fatalf("expected 2 elements, got %d", len(res))
	}
	if res[0] != 1.0 || res[1] != -2.0 {
		t.Errorf("unexpected values: %v", res)
	}
}

func TestDequantizeQ8_0_Kernel(t *testing.T) {
	// Block size 32. Q8_0 is 34 bytes (2-byte f16 delta + 32-byte int8).
	data := make([]byte, 34)
	// delta = 1.0 (0x3c00 in little endian)
	data[0] = 0x00
	data[1] = 0x3c
	for i := 0; i < 32; i++ {
		data[2+i] = byte(int8(i))
	}

	res := DequantizeQ8_0(data, 32)
	if len(res) != 32 {
		t.Fatalf("expected 32 elements, got %d", len(res))
	}
	if res[10] != 10.0 {
		t.Errorf("expected 10.0, got %f", res[10])
	}
}

func TestDequantizeQ4_0_Kernel(t *testing.T) {
	// Block size 32. Q4_0 is 18 bytes (2-byte f16 delta + 16-byte nibbles).
	data := make([]byte, 18)
	// delta = 2.0 (0x4000 in little endian)
	data[0] = 0x00
	data[1] = 0x40
	// 0x11 = (1, 1) in nibbles
	for i := 0; i < 16; i++ {
		data[2+i] = 0x11
	}

	res := DequantizeQ4_0(data, 32)
	if len(res) != 32 {
		t.Fatalf("expected 32 elements, got %d", len(res))
	}
	// q = (byte & 0x0F) - 8 = 1 - 8 = -7
	// val = delta * q = 2 * -7 = -14
	if res[0] != -14.0 {
		t.Errorf("expected -14.0, got %f", res[0])
	}
}

func TestDequantizeQ6K_Kernel(t *testing.T) {
	// Q6_K block: 128 bytes low nibbles + 64 bytes high bits
	//             + 16 int8 scales + 2 bytes f16 scale = 210 bytes.
	data := make([]byte, 210)
	for i := 0; i < 208; i++ {
		data[i] = 0xAA
	}
	binary.LittleEndian.PutUint16(data[208:210], Float32ToFloat16(1.5))

	res := DequantizeQ6K(data, 256)
	if len(res) != 256 {
		t.Fatalf("expected 256 elements, got %d", len(res))
	}

	// Every byte is 0xAA, so every value decodes the same way:
	//   low nibble 0xA, high bits (0xAA>>0)&3 == 2 -> raw 0x2A == 42
	//   scale int8(0xAA) == -86, d == 1.5 (exactly representable in f16)
	//   value == 1.5 * -86 * (42-32) == -1290
	const want = float32(-1290)
	for i, v := range res {
		if v != want {
			t.Fatalf("element %d: got %v, want %v", i, v, want)
		}
	}
}

func TestDequantizeBlock_Matrix(t *testing.T) {
	// Cover the generic wrapper
	data := []byte{0x00, 0x00, 0x00, 0x00}
	dst := make([]float32, 1)
	DequantizeBlock(data, dst, GGMLTypeF32)
	if dst[0] != 0 {
		t.Errorf("expected 0, got %f", dst[0])
	}
}
