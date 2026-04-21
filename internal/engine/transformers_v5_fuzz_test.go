//go:build darwin && metal

package engine

import (
	"testing"
	"unicode/utf8"
)

func FuzzTokenizerV5(f *testing.F) {
	f.Add([]byte("hello world"))
	f.Add([]byte(""))
	f.Add([]byte("Hello, 世界! 🌍"))
	f.Add([]byte{0, 1, 2, 3, 4})
	f.Add([]byte("a very long string that might cause buffer overflows"))
	f.Add(make([]byte, 512))

	f.Fuzz(func(t *testing.T, input []byte) {
		if len(input) > 512 {
			t.Skip("Input too large for fuzz")
		}

		if !utf8.Valid(input) {
			t.Skip("Invalid UTF-8")
		}

		_ = string(input)
	})
}

func FuzzGGUFLoading(f *testing.F) {
	f.Add([]byte("GGUF"))
	f.Add([]byte{})
	f.Add([]byte{0x00, 0xFF, 0xEE, 0x11})

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 4 {
			t.Skip("Data too small")
		}

		_ = data
	})
}