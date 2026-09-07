package vlm

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"golang.org/x/image/draw"
)

// VisionEncoder handles loading and executing multimodal CLIP/SigLIP transformer stacks.
type VisionEncoder struct {
	ctx          *device.Context
	dim          int
	weights      *VisionWeights
	Architecture string
	ImageMean    []float32
	ImageStd     []float32
}

type VisionWeights struct {
	PatchEmbed  *device.Tensor
	ProjectionB *device.Tensor
}

func NewVisionEncoder(ctx *device.Context, dim int, arch string) *VisionEncoder {
	mean := []float32{0.48145466, 0.4578275, 0.40821073}
	std := []float32{0.26862954, 0.26130259, 0.27577711}
	return &VisionEncoder{
		ctx:          ctx,
		dim:          dim,
		Architecture: arch,
		ImageMean:    mean,
		ImageStd:     std,
	}
}

func (v *VisionEncoder) SetWeights(w *VisionWeights) {
	v.weights = w
}

// Encode pixels into tensor representations compatible with LLM prefill stages.
func (v *VisionEncoder) Encode(imageData []byte) (*device.Tensor, error) {
	start := time.Now()
	if len(imageData) == 0 {
		return nil, fmt.Errorf("empty image payload")
	}

	reader := bytes.NewReader(imageData)
	img, _, err := image.Decode(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}

	const (
		TargetW  = 224
		TargetH  = 224
		Channels = 3
	)

	pixels := v.resizeAndNormalize(img, TargetW, TargetH, Channels)

	numPatches := (TargetW / 14) * (TargetH / 14)
	outTensor := v.ctx.NewTensorFP32(numPatches, v.dim)

	pixelTensor := v.ctx.NewTensorFP32(Channels, TargetW*TargetH)
	_ = pixelTensor.LoadFrom(pixels)

	var patchEmbed, projB *device.Tensor
	if v.weights != nil {
		patchEmbed = v.weights.PatchEmbed
		projB = v.weights.ProjectionB
	}
	if patchEmbed == nil {
		patchEmbed = v.ctx.NewTensorFP32(Channels*14*14, v.dim)
		defer patchEmbed.Free()
	}

	if v.Architecture == "gemma4" {
		v.ctx.VisionPatchEmbedGemma4(pixelTensor, patchEmbed, projB, outTensor, 14, v.dim, numPatches)
	} else {
		v.ctx.VisionPatchEmbed(pixelTensor, patchEmbed, outTensor, 14, v.dim, TargetW/14)
	}

	pixelTensor.Free()
	metrics.RecordVLMEncoding(numPatches, time.Since(start))
	return outTensor, nil
}

func (v *VisionEncoder) resizeAndNormalize(img image.Image, targetW, targetH, channels int) []float32 {
	bounds := img.Bounds()
	srcW := bounds.Dx()
	srcH := bounds.Dy()

	pixels := make([]float32, targetW*targetH*channels)

	mean := v.ImageMean
	std := v.ImageStd
	if mean == nil {
		mean = []float32{0.48145466, 0.4578275, 0.40821073}
	}
	if std == nil {
		std = []float32{0.26862954, 0.26130259, 0.27577711}
	}

	dst := image.NewNRGBA(image.Rect(0, 0, targetW, targetH))
	draw.CatmullRom.Scale(dst, dst.Rect, img, img.Bounds(), draw.Over, nil)

	for y := 0; y < targetH; y++ {
		for x := 0; x < targetW; x++ {
			r, g, b, _ := dst.At(x, y).RGBA()

			rFloat := float32(r>>8) / 255.0
			gFloat := float32(g>>8) / 255.0
			bFloat := float32(b>>8) / 255.0

			idx := (y*targetW + x) * channels
			pixels[idx] = (rFloat - mean[0]) / std[0]
			pixels[idx+1] = (gFloat - mean[1]) / std[1]
			pixels[idx+2] = (bFloat - mean[2]) / std[2]
		}
	}

	_ = srcW
	_ = srcH
	_ = color.RGBA{}

	return pixels
}
