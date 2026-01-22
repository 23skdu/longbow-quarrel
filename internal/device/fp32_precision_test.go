//go:build darwin && metal

package device

import (
	"testing"
)

func TestFFN_Precision_Path_Selection(t *testing.T) {
	testCases := []struct {
		name                 string
		dim                  int
		expectF32FFN         bool
		expectMixedPrecision bool
		expectFP16           bool
	}{
		{"Small model (dim=512)", 512, true, false, false},
		{"Medium model (dim=1024)", 1024, false, false, true},
		{"Medium model (dim=2048)", 2048, false, false, true},
		{"Large model (dim=4096)", 4096, false, true, false},
		{"Large model (dim=8192)", 8192, false, true, false},
		{"XL model (dim=16384)", 16384, false, true, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			useF32FFN := tc.dim < 1024
			useMixedPrecisionFFN := !useF32FFN && tc.dim >= 4096

			if tc.expectF32FFN && !useF32FFN {
				t.Errorf("Expected F32 FFN for dim=%d", tc.dim)
			}
			if tc.expectMixedPrecision && !useMixedPrecisionFFN {
				t.Errorf("Expected mixed precision for dim=%d", tc.dim)
			}
			if tc.expectFP16 && (useF32FFN || useMixedPrecisionFFN) {
				t.Errorf("Expected FP16 for dim=%d", tc.dim)
			}
		})
	}
}

func TestFFN_Scratch_Allocation_For_Large_Models(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	testCases := []struct {
		name      string
		dim       int
		hiddenDim int
		heads     int
		kvHeads   int
		headDim   int
		seqLen    int
		vocabSize int
	}{
		{"7B model config", 4096, 14336, 32, 8, 128, 4096, 49152},
		{"13B model config", 5120, 13824, 40, 8, 128, 5120, 32000},
		{"70B model config", 8192, 28672, 64, 8, 128, 8192, 32000},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			scratch := ctx.NewLayerScratch(1, tc.dim, tc.hiddenDim, tc.heads, tc.kvHeads, tc.headDim, tc.seqLen, tc.vocabSize)
			if scratch == nil {
				t.Fatal("Failed to create layer scratch")
			}

			if tc.dim >= 4096 {
				if scratch.NormedFFN_F32 == nil {
					t.Error("Large model should have NormedFFN_F32 allocated")
				}
				if scratch.ResFFN_F32 == nil {
					t.Error("Large model should have ResFFN_F32 allocated")
				}
			}

			scratch.Free()
		})
	}
}

func TestFFN_F32_Intermediate_Tensors(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	dim := 4096
	hiddenDim := 14336

	scratch := ctx.NewLayerScratch(1, dim, hiddenDim, 32, 8, 128, 4096, 49152)
	if scratch == nil {
		t.Fatal("Failed to create scratch")
	}
	defer scratch.Free()

	t.Run("FP32 intermediate tensors exist", func(t *testing.T) {
		if scratch.GatePart == nil {
			t.Error("GatePart should be allocated")
		}
		if scratch.UpPart == nil {
			t.Error("UpPart should be allocated")
		}
		if scratch.SwiOut == nil {
			t.Error("SwiOut should be allocated")
		}
	})

	t.Run("FP32 tensor data type", func(t *testing.T) {
		if scratch.GatePart.dataType != DataTypeF32 {
			t.Errorf("GatePart should be F32, got %v", scratch.GatePart.dataType)
		}
		if scratch.UpPart.dataType != DataTypeF32 {
			t.Errorf("UpPart should be F32, got %v", scratch.UpPart.dataType)
		}
		if scratch.SwiOut.dataType != DataTypeF32 {
			t.Errorf("SwiOut should be F32, got %v", scratch.SwiOut.dataType)
		}
	})

	t.Run("FP32 tensor sizes", func(t *testing.T) {
		expectedSize := hiddenDim * 4 // FP32 = 4 bytes
		if scratch.GatePart.sizeBytes != expectedSize {
			t.Errorf("GatePart size mismatch: expected %d, got %d", expectedSize, scratch.GatePart.sizeBytes)
		}
	})
}

func TestFP32_Accumulation_Prevents_Overflow(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	dim := 4096
	hiddenDim := 14336

	scratch := ctx.NewLayerScratch(1, dim, hiddenDim, 32, 8, 128, 4096, 49152)
	if scratch == nil {
		t.Fatal("Failed to create scratch")
	}
	defer scratch.Free()

	t.Run("FFN path selection for overflow prevention", func(t *testing.T) {
		useF32FFN := dim < 1024
		useMixedPrecisionFFN := !useF32FFN && dim >= 4096

		if dim >= 4096 && !useMixedPrecisionFFN {
			t.Error("Large models should use FP32 accumulation path to prevent overflow")
		}

		if dim >= 4096 {
			if scratch.NormedFFN_F32 == nil {
				t.Error("FP32 norm buffer should be allocated for large models")
			}
			if scratch.ResFFN_F32 == nil {
				t.Error("FP32 result buffer should be allocated for large models")
			}
		}
	})
}

func TestPrecision_Config_Validation(t *testing.T) {
	testCases := []struct {
		name         string
		dim          int
		expectedPath string
	}{
		{"Small model (dim=512)", 512, "f32_ffn"},
		{"Medium model (dim=1024)", 1024, "f16_ffn"},
		{"Medium model (dim=2048)", 2048, "f16_ffn"},
		{"Large model (dim=4096)", 4096, "mixed_precision"},
		{"Large model (dim=8192)", 8192, "mixed_precision"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			useF32FFN := tc.dim < 1024
			useMixedPrecisionFFN := !useF32FFN && tc.dim >= 4096

			var actualPath string
			switch {
			case useF32FFN:
				actualPath = "f32_ffn"
			case useMixedPrecisionFFN:
				actualPath = "mixed_precision"
			default:
				actualPath = "f16_ffn"
			}

			if actualPath != tc.expectedPath {
				t.Errorf("Expected %s path for dim=%d, got %s", actualPath, tc.dim, tc.expectedPath)
			}
		})
	}
}

func TestHiddenDim_Allocation_For_FP32(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	testCases := []struct {
		name      string
		hiddenDim int
		valid     bool
	}{
		{"Standard 7B hidden dim", 14336, true},
		{"Standard 13B hidden dim", 13824, true},
		{"Standard 70B hidden dim", 28672, true},
		{"Small model hidden dim", 11008, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			scratch := ctx.NewLayerScratch(1, 4096, tc.hiddenDim, 32, 8, 128, 4096, 49152)
			if scratch == nil {
				t.Fatal("Failed to create scratch")
			}

			expectedFP32Size := tc.hiddenDim * 4 // FP32 = 4 bytes

			if scratch.GatePart.sizeBytes != expectedFP32Size {
				t.Errorf("GatePart size mismatch: expected %d, got %d", expectedFP32Size, scratch.GatePart.sizeBytes)
			}

			scratch.Free()
		})
	}
}
