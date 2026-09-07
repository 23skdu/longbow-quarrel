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
