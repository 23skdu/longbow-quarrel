//go:build darwin && metal

package vlm

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

// VisionEncoder handles loading and executing multimodal CLIP/SigLIP transformer stacks.
type VisionEncoder struct {
	ctx     *device.Context
	dim     int
	weights *VisionWeights
}

type VisionWeights struct {
	PatchEmbed *device.Tensor
}

func NewVisionEncoder(ctx *device.Context, dim int) *VisionEncoder {
	return &VisionEncoder{
		ctx: ctx,
		dim: dim,
	}
}

// Encode pixels into tensor representations compatible with LLM prefill stages.
func (v *VisionEncoder) Encode(imageData []byte) (*device.Tensor, error) {
	if len(imageData) == 0 {
		return nil, fmt.Errorf("empty image payload")
	}

	reader := bytes.NewReader(imageData)
	_, _, err := image.Decode(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}

	// 1. Resize to target (e.g., 224x224)
	// For simplicity in the encoder logic, we assume a fixed patch size
	const (
		TargetW = 224
		TargetH = 224
		Channels = 3
	)

	// Stub: Actual high-quality bicubic resampling would happen here.
	// We'll normalize to [-1, 1] or [0, 1] as required by the model.
	pixels := make([]float32, TargetW*TargetH*Channels)
	
	// 2. Linear projection into LLM dimension
	numPatches := (TargetW / 14) * (TargetH / 14)
	outTensor := v.ctx.NewTensorFP32(numPatches, v.dim)
	
	// 3. Create GPU tensor for pixels
	pixelTensor := v.ctx.NewTensorFP32(Channels, TargetW*TargetH)
	pixelTensor.LoadFrom(pixels)
	
	// 4. Run CLIP Projection
	v.ctx.VisionPatchEmbed(pixelTensor, v.weights.PatchEmbed, outTensor, 14, v.dim, TargetW/14)
	
	pixelTensor.Free()
	return outTensor, nil
}
