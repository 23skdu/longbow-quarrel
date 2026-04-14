package engine

import (
	"os"
	"testing"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func TestGGUF_Error_Coverage(t *testing.T) {
	// 1. Invalid signature
	tmp, _ := os.CreateTemp("", "bad_sig.gguf")
	defer os.Remove(tmp.Name())
	tmp.Write([]byte("NOTGGUF"))
	tmp.Close()

	_, err := gguf.LoadFile(tmp.Name())
	if err == nil {
		t.Error("Expected error for invalid GGUF signature")
	}

	// 2. Short file
	tmp2, _ := os.CreateTemp("", "short.gguf")
	defer os.Remove(tmp2.Name())
	tmp2.Write([]byte("GGUF")) // Just the magic
	tmp2.Close()
	_, err = gguf.LoadFile(tmp2.Name())
	if err == nil {
		t.Error("Expected error for short GGUF file")
	}
}

func TestEngine_Metadata_Fallbacks(t *testing.T) {
	// These test the fallbacks in loadModel when KV is missing
	f := &gguf.GGUFFile{
		KV: make(map[string]interface{}),
	}
	
	// Use the unexported getKV helper if possible, or test via a minimal engine
	val, ok := getKV(f, "llama.block_count")
	if ok || val != nil {
		t.Error("Expected nil for missing key")
	}
}

func TestEngine_ToFloat64_Coverage(t *testing.T) {
	cases := []struct {
		in  interface{}
		out float64
	}{
		{int(42), 42.0},
		{uint32(42), 42.0},
		{uint64(42), 42.0},
		{float32(42.0), 42.0},
		{"not a number", 0.0},
		{nil, 0.0},
	}
	for _, c := range cases {
		if res := toFloat64(c.in); res != c.out {
			t.Errorf("toFloat64(%v) = %f, expected %f", c.in, res, c.out)
		}
	}
}
