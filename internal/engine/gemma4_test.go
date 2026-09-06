package engine

import (
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/simd"
)

func TestGemma4MultiTokenInference(t *testing.T) {
	modelPath := "/home/rsd/.cache/llmfit/models/Gemma4_E2B_Abliterated_Opus_Distilled.Q8_0.gguf"
	f, err := gguf.LoadFile(modelPath)
	if err != nil {
		t.Skipf("Gemma 4 model not found: %v", err)
		return
	}
	defer f.Close()

	modelCfg := ExtractModelConfig(f)

	cfg := config.Config{
		Dim:                   modelCfg.Dim,
		Heads:                 modelCfg.Heads,
		KVHeads:               modelCfg.KVHeads,
		Layers:                modelCfg.Layers,
		VocabSize:             modelCfg.VocabSize,
		HeadDim:               modelCfg.HeadDim,
		IsGemma4:              true,
		FinalLogitSoftcapping: 30.0,
		Eps:                   1e-6,
	}

	w, err := loadCPUWeights(f, cfg)
	if err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	// Prompt: "The capital of France is" -> tokens: [2, 818, 5279, 529, 7001, 563]
	tokens := []int{2, 818, 5279, 529, 7001, 563}
	kv := &CPUKVCache{}

	var lastHidden []float32
	for pos, tok := range tokens {
		emb := w.GetTokenEmbedding(tok, cfg.Dim)
		scaleEmb := float32(math.Sqrt(float64(cfg.Dim)))
		hidden := make([]float32, len(emb))
		for i, v := range emb {
			hidden[i] = v * scaleEmb
		}

		ple := w.ComputeGemma4PLE(tok, hidden, cfg.Layers)
		for layerIdx := 0; layerIdx < cfg.Layers; layerIdx++ {
			hidden = ApplyGemma4LayerCPU(w, hidden, layerIdx, pos, kv, ple[layerIdx], cfg)
		}
		lastHidden = hidden
	}

	// Output norm
	normed := make([]float32, len(lastHidden))
	simd.RMSNorm(lastHidden, w.OutputNorm, normed, 1, len(lastHidden), cfg.Eps)

	// Multiply by token_embd.weight (tied embeddings)
	logits := w.MatVec(w.Output, w.RawOutput, normed)
	for j := range logits {
		logits[j] = 30.0 * float32(math.Tanh(float64(logits[j]/30.0)))
	}

	type pair struct {
		id    int
		logit float32
	}
	top := make([]pair, 5)
	for i, l := range logits {
		for k := 0; k < len(top); k++ {
			if l > top[k].logit {
				copy(top[k+1:], top[k:])
				top[k] = pair{id: i, logit: l}
				break
			}
		}
	}

	for rank, p := range top {
		t.Logf("Rank %d: token %d, logit = %f", rank+1, p.id, p.logit)
	}

	// Expected token 9079 (" Paris") with top rank
	if top[0].id != 9079 {
		t.Errorf("Expected top token 9079 (Paris), got %d (logit %f)", top[0].id, top[0].logit)
	}
	expectedLogit := float32(11.9683)
	if diff := float32(math.Abs(float64(top[0].logit - expectedLogit))); diff > 0.5 {
		t.Errorf("Expected top logit ~%f, got %f (diff %f)", expectedLogit, top[0].logit, diff)
	}
}
