package vlm

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

func createTestPNG(width, height int) []byte {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, color.RGBA{R: uint8(x % 255), G: uint8(y % 255), B: 128, A: 255})
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	return buf.Bytes()
}

func TestVisionEncoder_Encode(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	hiddenDim := 64
	encoder := NewVisionEncoder(ctx, hiddenDim, "clip")

	pngBytes := createTestPNG(32, 32)
	tensor, err := encoder.Encode(pngBytes)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}
	defer tensor.Free()

	// 224/14 = 16, 16*16 = 256 patches
	expectedPatches := (224 / 14) * (224 / 14)
	if tensor.Rows() != expectedPatches {
		t.Errorf("expected %d rows (patches), got %d", expectedPatches, tensor.Rows())
	}
	if tensor.Cols() != hiddenDim {
		t.Errorf("expected %d cols, got %d", hiddenDim, tensor.Cols())
	}
}

func TestVisionEncoder_Gemma4(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	hiddenDim := 64
	encoder := NewVisionEncoder(ctx, hiddenDim, "gemma4")

	weights := &VisionWeights{
		PatchEmbed:  ctx.NewTensorFP32(3*14*14, hiddenDim),
		ProjectionB: ctx.NewTensorFP32(1, hiddenDim),
	}
	defer weights.PatchEmbed.Free()
	defer weights.ProjectionB.Free()
	encoder.SetWeights(weights)

	pngBytes := createTestPNG(48, 48)
	tensor, err := encoder.Encode(pngBytes)
	if err != nil {
		t.Fatalf("Encode Gemma4 failed: %v", err)
	}
	defer tensor.Free()

	expectedPatches := (224 / 14) * (224 / 14)
	if tensor.Rows() != expectedPatches {
		t.Errorf("expected %d rows, got %d", expectedPatches, tensor.Rows())
	}
}

func TestVLMDecoder_Lifecycle(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	cfg := VLMConfig{
		Architecture: "clip",
		ImageSize:    224,
		PatchSize:    14,
		HiddenDim:    64,
		NumLayers:    2,
	}

	dec, err := NewVLMDecoder(ctx, cfg)
	if err != nil {
		t.Fatalf("NewVLMDecoder failed: %v", err)
	}

	pngBytes := createTestPNG(16, 16)
	tensor, err := dec.Decode(pngBytes)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	defer tensor.Free()

	if tensor.Rows() != 256 {
		t.Errorf("expected 256 patches, got %d", tensor.Rows())
	}
}

func TestVLM_CoverageExhaustive(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	pngBytes := createTestPNG(16, 16)

	// 1. Architectures in NewVLMDecoder
	archs := []string{"siglip", "gemma4", "llava", "qwen-vl"}
	for _, arch := range archs {
		cfg := VLMConfig{
			Architecture: arch,
			ImageSize:    224,
			PatchSize:    14,
			HiddenDim:    64,
		}
		dec, err := NewVLMDecoder(ctx, cfg)
		if err != nil {
			t.Fatalf("NewVLMDecoder failed for %s: %v", arch, err)
		}
		tensor, err := dec.Decode(pngBytes)
		if err != nil {
			t.Fatalf("Decode failed for %s: %v", arch, err)
		}
		tensor.Free()
	}

	// Unsupported architecture
	badCfg := VLMConfig{Architecture: "unsupported"}
	if _, err := NewVLMDecoder(ctx, badCfg); err == nil {
		t.Error("expected error for unsupported VLM architecture")
	}

	// 2. MultiModalEncoder.Decode error path on invalid image
	cfgLlava := VLMConfig{
		Architecture: "llava",
		ImageSize:    224,
		PatchSize:    14,
		HiddenDim:    64,
	}
	llavaDec, err := NewVLMDecoder(ctx, cfgLlava)
	if err != nil {
		t.Fatalf("failed to create llava decoder: %v", err)
	}
	if _, err := llavaDec.Decode([]byte("corrupt data")); err == nil {
		t.Error("expected error decoding corrupt image with MultiModalEncoder")
	}

	// 3. VisionEncoder error paths
	enc := NewVisionEncoder(ctx, 64, "clip")
	// Empty payload
	if _, err := enc.Encode(nil); err == nil {
		t.Error("expected error for empty image payload")
	}
	// Corrupt payload
	if _, err := enc.Encode([]byte("invalid image")); err == nil {
		t.Error("expected error for corrupt image payload")
	}

	// 4. Custom ImageMean and ImageStd
	enc.ImageMean = []float32{0.5, 0.5, 0.5}
	enc.ImageStd = []float32{0.25, 0.25, 0.25}
	tensorCustom, err := enc.Encode(pngBytes)
	if err != nil {
		t.Fatalf("Encode with custom mean/std failed: %v", err)
	}
	tensorCustom.Free()

	// 5. Nil ImageMean and ImageStd fallback
	enc.ImageMean = nil
	enc.ImageStd = nil
	tensorNilMean, err := enc.Encode(pngBytes)
	if err != nil {
		t.Fatalf("Encode with nil mean/std failed: %v", err)
	}
	tensorNilMean.Free()
}

