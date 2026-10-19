// Command quantize converts a GGUF model into a quantized GGUF model.
//
// The quantization functions in internal/gguf existed, but there was no way
// to use them: the repository could read GGUF files and not write them, so
// a quantized model could not be produced at all.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

var (
	inputPath  = flag.String("i", "", "Path to the input GGUF model")
	outputPath = flag.String("o", "", "Path to the output GGUF model")
	targetType = flag.String("t", "q4_k", "Target quantization: q4_k, q6_k, q8_0, f16, f32")
	keepEmbed  = flag.Bool("keep-embeddings", true, "Keep the token embedding in float16 instead of quantizing it")
	dryRun     = flag.Bool("dry-run", false, "Report what would be done without writing the output")
)

func main() {
	flag.Parse()

	if *inputPath == "" || *outputPath == "" {
		fmt.Fprintln(os.Stderr, "usage: quantize -i <input.gguf> -o <output.gguf> [-t q4_k|q6_k|q8_0|f16|f32]")
		flag.PrintDefaults()
		os.Exit(2)
	}

	if err := run(*inputPath, *outputPath, *targetType, *keepEmbed, *dryRun); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(inputPath, outputPath, targetName string, keepEmbed, dryRun bool) error {
	target, err := parseTarget(targetName)
	if err != nil {
		return err
	}

	model, err := gguf.LoadFile(inputPath)
	if err != nil {
		return fmt.Errorf("failed to load %s: %w", inputPath, err)
	}
	defer func() { _ = model.Close() }()

	fmt.Printf("Loaded %s: %d tensors, %d metadata entries\n",
		inputPath, len(model.Tensors), len(model.KV))

	out := make([]gguf.TensorData, 0, len(model.Tensors))
	quantized := 0
	copied := 0
	var before, after int64

	for _, t := range model.Tensors {
		before += int64(t.SizeBytes())

		// The token embedding is usually kept at a higher precision: it
		// is small, and it is the tensor whose errors are the most
		// visible in the output.
		isEmbedding := strings.Contains(t.Name, "token_embd") || strings.Contains(t.Name, "embed_tokens")

		if isEmbedding && keepEmbed {
			out = append(out, gguf.TensorData{
				Name:       t.Name,
				Dimensions: t.Dimensions,
				Type:       gguf.GGMLTypeF16,
				Data:       toF16(t, mustDequantize(t)),
			})
			after += int64(len(out[len(out)-1].Data))
			copied++
			continue
		}

		values, err := gguf.DequantizeTensor(t)
		if err != nil {
			// A tensor that cannot be dequantized is copied verbatim
			// rather than dropped: a model with one unsupported tensor
			// is still usable, a model with a missing tensor is not.
			fmt.Fprintf(os.Stderr, "warning: %s: %v (copied verbatim)\n", t.Name, err)
			out = append(out, gguf.TensorData{
				Name:       t.Name,
				Dimensions: t.Dimensions,
				Type:       t.Type,
				Data:       trimTo(t, t.SizeBytes()),
			})
			after += int64(len(out[len(out)-1].Data))
			copied++
			continue
		}

		data, newType, err := quantizeTo(target, values, t)
		if err != nil {
			return fmt.Errorf("failed to quantize %s: %w", t.Name, err)
		}

		out = append(out, gguf.TensorData{
			Name:       t.Name,
			Dimensions: t.Dimensions,
			Type:       newType,
			Data:       data,
		})
		after += int64(len(data))
		quantized++
	}

	fmt.Printf("Quantized %d tensors, copied %d\n", quantized, copied)
	fmt.Printf("Size: %.1f MB -> %.1f MB (%.1f%%)\n",
		float64(before)/(1024*1024), float64(after)/(1024*1024),
		100*float64(after)/float64(before))

	if dryRun {
		fmt.Println("Dry run: no file was written")
		return nil
	}

	if err := gguf.WriteFile(outputPath, model.KV, out, alignmentOf(model)); err != nil {
		return fmt.Errorf("failed to write %s: %w", outputPath, err)
	}
	fmt.Printf("Wrote %s\n", outputPath)

	return nil
}

// target describes a quantization format.
type target struct {
	name string
	typ  gguf.GGMLType
}

func parseTarget(name string) (target, error) {
	switch strings.ToLower(name) {
	case "q4_k":
		return target{name: "Q4_K", typ: gguf.GGMLTypeQ4_K}, nil
	case "q6_k":
		return target{name: "Q6_K", typ: gguf.GGMLTypeQ6_K}, nil
	case "q8_0":
		return target{name: "Q8_0", typ: gguf.GGMLTypeQ8_0}, nil
	case "f32":
		return target{name: "F32", typ: gguf.GGMLTypeF32}, nil
	case "f16":
		return target{name: "F16", typ: gguf.GGMLTypeF16}, nil
	default:
		return target{}, fmt.Errorf("unknown target type %q", name)
	}
}

// quantizeTo encodes the values in the target format.
func quantizeTo(t target, values []float32, info *gguf.TensorInfo) ([]byte, gguf.GGMLType, error) {
	switch t.typ {
	case gguf.GGMLTypeQ4_K:
		data, err := gguf.QuantizeWeightsToQ4K(values, len(values))
		if err != nil {
			return nil, 0, err
		}
		return data, t.typ, nil
	case gguf.GGMLTypeQ6_K:
		data := gguf.QuantizeQ6K(values)
		return data, t.typ, nil
	case gguf.GGMLTypeQ8_0:
		data, err := gguf.QuantizeWeightsToQ8_0(values, len(values))
		if err != nil {
			return nil, 0, err
		}
		return data, t.typ, nil
	case gguf.GGMLTypeF32:
		out := make([]byte, len(values)*4)
		for i, v := range values {
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
		}
		return out, t.typ, nil
	case gguf.GGMLTypeF16:
		return toF16(info, values), t.typ, nil
	default:
		return nil, 0, fmt.Errorf("unsupported target type %v", t.typ)
	}
}

// toF16 encodes the values in half precision.
func toF16(info *gguf.TensorInfo, values []float32) []byte {
	out := make([]byte, len(values)*2)
	for i, v := range values {
		binary.LittleEndian.PutUint16(out[i*2:], gguf.Float32ToFloat16(v))
	}
	return out
}

// mustDequantize dequantizes a tensor, returning nil if it cannot.
func mustDequantize(t *gguf.TensorInfo) []float32 {
	values, err := gguf.DequantizeTensor(t)
	if err != nil {
		return nil
	}
	return values
}

// trimTo returns the first n bytes of the tensor's data.
func trimTo(t *gguf.TensorInfo, n uint64) []byte {
	if uint64(len(t.Data)) <= n {
		return t.Data
	}
	return t.Data[:n]
}

// alignmentOf returns the tensor alignment the model asked for.
func alignmentOf(model *gguf.GGUFFile) uint64 {
	if v, ok := model.KV["general.alignment"].(uint32); ok && v > 0 {
		return uint64(v)
	}
	if v, ok := model.KV["general.alignment"].(uint64); ok && v > 0 {
		return v
	}
	return gguf.GGUFDefaultAlignment
}
