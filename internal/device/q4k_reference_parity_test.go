package device

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// TestQ4KDecoderMatchesTheReference compares the production Q4_K decoder
// with the reference implementation in this file.
//
// The reference is llama.cpp's format, written out in Go. It was here but
// nothing called it, so the decoder that ships was never checked against
// the format it claims to implement.
func TestQ4KDecoderMatchesTheReference(t *testing.T) {
	// A block of 144 bytes holds 256 values.
	const (
		blockSize  = 144
		numValues  = 256
		numBlocks  = 8
		blockBytes = blockSize * numBlocks
	)

	// The block is built the way the format says: two half-precision
	// scales, twelve bytes of packed six-bit scales and 128 bytes of
	// four-bit quants.
	block := make([]byte, blockBytes)
	for i := range block {
		// A deterministic but not uniform pattern, so that a decoder that
		// reads the wrong field produces a different answer.
		block[i] = byte((i*37 + 11) % 251)
	}
	// The two half-precision scales: a small positive one and a small
	// positive one, so the dequantized values are not dominated by one
	// of them.
	binary.LittleEndian.PutUint16(block[0:2], gguf.Float32ToFloat16(0.5))
	binary.LittleEndian.PutUint16(block[2:4], gguf.Float32ToFloat16(0.25))

	want := DequantizeQ4K_Reference(block[:blockSize])

	got := gguf.DequantizeQ4K(block[:blockSize], numValues)
	if len(got) != numValues {
		t.Fatalf("the decoder returned %d values, want %d", len(got), numValues)
	}

	// The scale of the values sets the tolerance: the two decoders round
	// the six-bit scales the same way, so they have to agree closely.
	scale := 0.0
	for _, v := range want {
		if a := math.Abs(float64(v)); a > scale {
			scale = a
		}
	}
	if scale == 0 {
		scale = 1
	}
	tolerance := scale * 1e-5

	for i := range want {
		if math.IsNaN(float64(got[i])) {
			t.Fatalf("the value %d is a NaN", i)
		}
		if diff := math.Abs(float64(got[i] - want[i])); diff > tolerance {
			t.Fatalf("the value %d is %v, the reference says %v (difference %v, tolerance %v)",
				i, got[i], want[i], diff, tolerance)
		}
	}
}
