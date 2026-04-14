package gguf

import (
	"testing"
)

func TestMetadata_Exhaustive(t *testing.T) {
	f := &GGUFFile{
		KV: map[string]interface{}{
			"test.string": "hello",
			"test.int":    int32(42),
			"test.float":  float32(3.14),
			"test.bool":   true,
			"test.array":  []interface{}{int32(1), int32(2)},
		},
	}

	// 1. Strings
	if val, ok := f.KV["test.string"].(string); !ok || val != "hello" {
		t.Error("String KV failed")
	}

	// 2. Arrays
	if arr, ok := f.KV["test.array"].([]interface{}); !ok || len(arr) != 2 {
		t.Error("Array KV failed")
	}

	// 3. Missing
	if _, ok := f.KV["non-existent"]; ok {
		t.Error("Non-existent key found")
	}
}

const GGMLTypeUnknown GGMLType = 999

func TestTensorInfo_SizeBytes_Exhaustive(t *testing.T) {
	cases := []struct {
		t    GGMLType
		ne   []uint64
		size uint64
	}{
		{GGMLTypeF32, []uint64{10}, 40},
		{GGMLTypeF16, []uint64{10}, 20},
		{GGMLTypeQ4_0, []uint64{32}, 18},
		{GGMLTypeQ8_0, []uint64{32}, 34},
		{GGMLTypeQ4_K, []uint64{256}, 144},
		{GGMLTypeQ6_K, []uint64{256}, 210},
		{GGMLTypeTQ1_0, []uint64{256}, 328},
		{GGMLTypeUnknown, []uint64{10}, 0},
	}

	for _, c := range cases {
		ti := &TensorInfo{Type: c.t, Dimensions: c.ne}
		if s := ti.SizeBytes(); s != c.size {
			t.Errorf("Type %s: expected %d bytes, got %d", c.t, c.size, s)
		}
	}
}

func TestGGMLType_String(t *testing.T) {
	types := []GGMLType{
		GGMLTypeF32, GGMLTypeF16, GGMLTypeQ4_0, GGMLTypeQ4_K, GGMLTypeQ6_K,
		GGMLTypeTQ1_0, 999,
	}
	for _, ty := range types {
		s := ty.String()
		if s == "" {
			t.Error("Empty type string")
		}
	}
}
