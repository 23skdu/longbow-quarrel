//go:build darwin && metal

package vlm

import (
	"fmt"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

type VLMDecoder interface {
	Decode(imageData []byte) (*device.Tensor, error)
}

type VLMConfig struct {
	Architecture string
	ImageSize    int
	PatchSize    int
	HiddenDim    int
	NumLayers    int
}

func NewVLMDecoder(ctx *device.Context, cfg VLMConfig) (VLMDecoder, error) {
	switch cfg.Architecture {
	case "clip", "siglip":
		return newCLIPEncoder(ctx, cfg), nil
	case "llava", "qwen-vl":
		return newMultiModalEncoder(ctx, cfg), nil
	default:
		return nil, fmt.Errorf("unsupported VLM architecture: %s", cfg.Architecture)
	}
}

type CLIPEncoder struct {
	ctx          *device.Context
	config       VLMConfig
	visionEncoder *VisionEncoder
}

func newCLIPEncoder(ctx *device.Context, cfg VLMConfig) *CLIPEncoder {
	enc := &CLIPEncoder{
		ctx:    ctx,
		config: cfg,
	}
	enc.visionEncoder = NewVisionEncoder(ctx, cfg.HiddenDim, cfg.Architecture)
	return enc
}

func (e *CLIPEncoder) Decode(imageData []byte) (*device.Tensor, error) {
	return e.visionEncoder.Encode(imageData)
}

type MultiModalEncoder struct {
	ctx          *device.Context
	config       VLMConfig
	visionEncoder *VisionEncoder
}

func newMultiModalEncoder(ctx *device.Context, cfg VLMConfig) *MultiModalEncoder {
	enc := &MultiModalEncoder{
		ctx:    ctx,
		config: cfg,
	}
	enc.visionEncoder = NewVisionEncoder(ctx, cfg.HiddenDim, cfg.Architecture)
	return enc
}

func (e *MultiModalEncoder) Decode(imageData []byte) (*device.Tensor, error) {
	tensor, err := e.visionEncoder.Encode(imageData)
	if err != nil {
		return nil, err
	}

	projected := e.ctx.NewTensor(tensor.Rows(), e.config.HiddenDim)
	e.ctx.VisionPatchEmbed(tensor, nil, projected, e.config.PatchSize, e.config.HiddenDim, tensor.Rows())

	return projected, nil
}