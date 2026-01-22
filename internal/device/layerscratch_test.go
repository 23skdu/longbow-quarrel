//go:build darwin && metal

package device

import (
	"testing"
)

func TestLayerScratch_Heap_Allocation_And_Free(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	scratch := ctx.NewLayerScratch(1, 4096, 14336, 32, 8, 128, 4096, 49152)
	if scratch == nil {
		t.Fatal("Failed to create LayerScratch")
	}

	if scratch.heap == nil {
		t.Error("Heap should be allocated")
	}

	scratch.Free()

	if scratch.heap != nil {
		t.Error("Heap should be nil after Free()")
	}
}

func TestLayerScratch_Multiple_Allocations(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	var scratches []*LayerScratch

	for i := 0; i < 5; i++ {
		scratch := ctx.NewLayerScratch(1, 4096, 14336, 32, 8, 128, 4096, 49152)
		if scratch == nil {
			t.Fatalf("Failed to create LayerScratch %d", i)
		}
		if scratch.heap == nil {
			t.Errorf("Heap should be allocated for scratch %d", i)
		}
		scratches = append(scratches, scratch)
	}

	for i, scratch := range scratches {
		scratch.Free()
		if scratch.heap != nil {
			t.Errorf("Heap should be nil after Free() for scratch %d", i)
		}
	}
}

func TestLayerScratch_Different_Configurations(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	configs := []struct {
		name      string
		batch     int
		dim       int
		hiddenDim int
		heads     int
		kvHeads   int
		headDim   int
		seqLen    int
		vocabSize int
	}{
		{"Small model (dim=512)", 1, 512, 2048, 8, 8, 64, 512, 32000},
		{"Medium model (dim=2048)", 1, 2048, 8192, 16, 16, 128, 2048, 32000},
		{"Large model (dim=4096)", 1, 4096, 14336, 32, 8, 128, 4096, 49152},
		{"XL model (dim=8192)", 1, 8192, 28672, 64, 8, 128, 8192, 32000},
		{"Batch 4 model", 4, 4096, 14336, 32, 8, 128, 4096, 49152},
	}

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			scratch := ctx.NewLayerScratch(cfg.batch, cfg.dim, cfg.hiddenDim, cfg.heads, cfg.kvHeads, cfg.headDim, cfg.seqLen, cfg.vocabSize)
			if scratch == nil {
				t.Fatal("Failed to create LayerScratch")
			}
			if scratch.heap == nil {
				t.Error("Heap should be allocated")
			}

			if scratch.QPart == nil {
				t.Error("QPart should be allocated")
			}
			if scratch.KPart == nil {
				t.Error("KPart should be allocated")
			}
			if scratch.VPart == nil {
				t.Error("VPart should be allocated")
			}
			if scratch.Scores == nil {
				t.Error("Scores should be allocated")
			}
			if scratch.GatePart == nil {
				t.Error("GatePart should be allocated")
			}
			if scratch.UpPart == nil {
				t.Error("UpPart should be allocated")
			}
			if scratch.Logits == nil {
				t.Error("Logits should be allocated")
			}

			scratch.Free()

			if scratch.heap != nil {
				t.Error("Heap should be nil after Free()")
			}
		})
	}
}

func TestLayerScratch_FP32_Buffers_Allocated(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	scratch := ctx.NewLayerScratch(1, 4096, 14336, 32, 8, 128, 4096, 49152)
	if scratch == nil {
		t.Fatal("Failed to create LayerScratch")
	}
	defer scratch.Free()

	if scratch.NormedFFN_F32 == nil {
		t.Error("NormedFFN_F32 should be allocated for large models")
	}
	if scratch.ResFFN_F32 == nil {
		t.Error("ResFFN_F32 should be allocated for large models")
	}
	if scratch.GatePart == nil {
		t.Error("GatePart should be allocated")
	}
	if scratch.UpPart == nil {
		t.Error("UpPart should be allocated")
	}
	if scratch.SwiOut == nil {
		t.Error("SwiOut should be allocated")
	}
}

func TestLayerScratch_Small_Model_FP32(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	scratch := ctx.NewLayerScratch(1, 512, 2048, 8, 8, 64, 512, 32000)
	if scratch == nil {
		t.Fatal("Failed to create LayerScratch")
	}
	defer scratch.Free()

	if scratch.GatePart == nil {
		t.Error("GatePart should be allocated")
	}
	if scratch.UpPart == nil {
		t.Error("UpPart should be allocated")
	}
	if scratch.SwiOut == nil {
		t.Error("SwiOut should be allocated")
	}
}

func TestLayerScratch_Free_Safety(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	scratch := ctx.NewLayerScratch(1, 4096, 14336, 32, 8, 128, 4096, 49152)
	if scratch == nil {
		t.Fatal("Failed to create LayerScratch")
	}

	scratch.Free()
	scratch.Free()
	scratch.Free()

	if scratch.heap != nil {
		t.Error("Heap should be nil after multiple Free() calls")
	}
}

func TestLayerScratch_Tensor_Ownership(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	scratch := ctx.NewLayerScratch(1, 4096, 14336, 32, 8, 128, 4096, 49152)
	if scratch == nil {
		t.Fatal("Failed to create LayerScratch")
	}

	allTensors := []*Tensor{
		scratch.QPart, scratch.KPart, scratch.VPart,
		scratch.AttOut, scratch.ResAtt, scratch.Scores,
		scratch.Normed, scratch.NormedFFN, scratch.NormedFFN_F32,
		scratch.ResFFN, scratch.ResFFN_F32,
		scratch.GatePart, scratch.UpPart, scratch.SwiOut, scratch.Logits,
	}

	for i, tensor := range allTensors {
		if tensor == nil {
			t.Errorf("Tensor %d should not be nil", i)
			continue
		}
		if tensor.buf == nil {
			t.Errorf("Tensor %d buffer should not be nil", i)
		}
	}

	scratch.Free()

	for i, tensor := range allTensors {
		if tensor.buf != nil {
			t.Errorf("Tensor %d buffer should be nil after Free()", i)
		}
	}
}

func TestLayerScratch_Heap_Size_Calculation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	testCases := []struct {
		name      string
		batch     int
		dim       int
		hiddenDim int
		heads     int
		kvHeads   int
		headDim   int
		seqLen    int
		vocabSize int
		minBytes  int
	}{
		{"Small model", 1, 512, 2048, 8, 8, 64, 512, 32000, 500000},
		{"Medium model", 1, 2048, 8192, 16, 16, 128, 2048, 32000, 2000000},
		{"Large model", 1, 4096, 14336, 32, 8, 128, 4096, 49152, 4000000},
		{"XL model", 1, 8192, 28672, 64, 8, 128, 8192, 32000, 8000000},
		{"Batch 4 large", 4, 4096, 14336, 32, 8, 128, 4096, 49152, 16000000},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			scratch := ctx.NewLayerScratch(tc.batch, tc.dim, tc.hiddenDim, tc.heads, tc.kvHeads, tc.headDim, tc.seqLen, tc.vocabSize)
			if scratch == nil {
				t.Fatal("Failed to create LayerScratch")
			}

			if scratch.heap == nil {
				t.Error("Heap should be allocated")
			}

			scratch.Free()
		})
	}
}
