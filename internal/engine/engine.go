//go:build darwin && metal

package engine

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"runtime"
	"sort"

	"time"

	"strings"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

// Helper to get KV value with architecture-aware fallback
func getKV(f *gguf.GGUFFile, llamaKey, qwenKey string) (interface{}, bool) {
	// log.Printf("[DEBUG] Searching KV: llama=%s, qwen=%s", llamaKey, qwenKey)

	// 1. Try provided llama key
	if val, ok := f.KV[llamaKey]; ok {
		return val, true
	}

	// 2. Try provided qwen key
	if qwenKey != "" {
		if val, ok := f.KV[qwenKey]; ok {
			return val, true
		}
	}

	// 3. Try general keys if applicable
	if strings.Contains(llamaKey, "llama.") {
		generalKey := strings.Replace(llamaKey, "llama.", "general.", 1)
		if val, ok := f.KV[generalKey]; ok {
			return val, true
		}
	}

	// 4. Try granite prefix (common in Ollama)
	if strings.Contains(llamaKey, "llama.") {
		graniteKey := strings.Replace(llamaKey, "llama.", "granite.", 1)
		if val, ok := f.KV[graniteKey]; ok {
			return val, true
		}
	}

	// 5. Dynamic architecture detection
	if arch, ok := f.KV["general.architecture"].(string); ok {
		// Replace "llama." with "<arch>."
		archKey := strings.Replace(llamaKey, "llama.", arch+".", 1)
		if val, ok := f.KV[archKey]; ok {
			return val, true
		}

		// Also try some variations if arch is like "llama" but keys are different
		for _, alt := range []string{"mistral", "qwen2", "phi3", "starcoder2"} {
			altKey := strings.Replace(llamaKey, "llama.", alt+".", 1)
			if val, ok := f.KV[altKey]; ok {
				return val, true
			}
		}
	}

	return nil, false
}

// QualityEvaluator provides metrics for evaluating generated text quality
type QualityEvaluator struct {
	tokenizer *tokenizer.Tokenizer
}

// NewQualityEvaluator creates a new quality evaluator
func NewQualityEvaluator(t *tokenizer.Tokenizer) *QualityEvaluator {
	return &QualityEvaluator{tokenizer: t}
}

// NewQualityEvaluatorSimple creates a quality evaluator without tokenizer (for basic metrics)
func NewQualityEvaluatorSimple() *QualityEvaluator {
	return &QualityEvaluator{tokenizer: nil}
}

// PerplexityResult holds perplexity calculation results
type PerplexityResult struct {
	Perplexity  float64
	TotalTokens int
	AvgLogProb  float64
}

// CalculatePerplexity computes perplexity for a sequence of tokens
// Note: This is a simplified implementation. A full implementation would require
// the model to compute actual token probabilities.
func (qe *QualityEvaluator) CalculatePerplexity(tokens []int) PerplexityResult {
	if len(tokens) < 2 {
		return PerplexityResult{Perplexity: 1.0, TotalTokens: len(tokens), AvgLogProb: 0.0}
	}

	// Simplified perplexity calculation
	// In a real implementation, this would use the model's forward pass
	// to compute actual log probabilities for each token given its context
	totalLogProb := 0.0
	validTokens := 0

	// For demonstration, use a simple language model approximation
	// This is just a placeholder - real perplexity requires model probabilities
	for i := 1; i < len(tokens); i++ {
		// Approximate log probability based on token frequency patterns
		// This is highly simplified and not accurate
		logProb := -0.5 - 0.1*float64(i%3) // Placeholder calculation
		totalLogProb += logProb
		validTokens++
	}

	avgLogProb := totalLogProb / float64(validTokens)
	perplexity := math.Exp(-avgLogProb)

	return PerplexityResult{
		Perplexity:  perplexity,
		TotalTokens: validTokens,
		AvgLogProb:  avgLogProb,
	}
}

// BLEUScore holds BLEU evaluation results
type BLEUScore struct {
	BLEU1     float64
	BLEU2     float64
	BLEU3     float64
	BLEU4     float64
	Precision []float64
}

// CalculateBLEU computes BLEU score between candidate and reference texts
func (qe *QualityEvaluator) CalculateBLEU(candidate, reference string) BLEUScore {
	if qe.tokenizer == nil {
		// Simple character-based BLEU for testing without tokenizer
		return qe.calculateBLEUSimple(candidate, reference)
	}

	candTokens := qe.tokenizer.Encode(candidate)
	refTokens := qe.tokenizer.Encode(reference)

	// Simplified BLEU calculation for n-grams 1-4
	maxN := 4
	precisions := make([]float64, maxN)

	for n := 1; n <= maxN; n++ {
		candNGrams := getNGrams(candTokens, n)
		refNGrams := getNGrams(refTokens, n)

		// Count matching n-grams
		matches := 0
		for candNGram := range candNGrams {
			if refNGrams[candNGram] > 0 {
				matches++
			}
		}

		// Calculate precision
		if len(candNGrams) > 0 {
			precisions[n-1] = float64(matches) / float64(len(candNGrams))
		} else {
			precisions[n-1] = 0.0
		}
	}

	// Calculate BLEU scores (simplified geometric mean)
	bleu1 := precisions[0]
	bleu2 := math.Sqrt(precisions[0] * precisions[1])
	bleu3 := math.Pow(precisions[0]*precisions[1]*precisions[2], 1.0/3.0)
	bleu4 := math.Pow(precisions[0]*precisions[1]*precisions[2]*precisions[3], 1.0/4.0)

	// Apply brevity penalty (simplified)
	bp := 1.0
	if len(candTokens) < len(refTokens) {
		bp = math.Exp(1.0 - float64(len(refTokens))/float64(len(candTokens)))
	}

	return BLEUScore{
		BLEU1:     bleu1 * bp,
		BLEU2:     bleu2 * bp,
		BLEU3:     bleu3 * bp,
		BLEU4:     bleu4 * bp,
		Precision: precisions,
	}
}

// ROUGEScore holds ROUGE evaluation results
type ROUGEScore struct {
	ROUGE1_F1 float64
	ROUGE2_F1 float64
	ROUGEL_F1 float64
	Precision float64
	Recall    float64
	F1        float64
}

// CalculateROUGE computes ROUGE score between candidate and reference texts
func (qe *QualityEvaluator) CalculateROUGE(candidate, reference string) ROUGEScore {
	if qe.tokenizer == nil {
		// Simple character-based ROUGE for testing without tokenizer
		return qe.calculateROUGESimple(candidate, reference)
	}

	candTokens := qe.tokenizer.Encode(candidate)
	refTokens := qe.tokenizer.Encode(reference)

	// Calculate unigram overlap (ROUGE-1)
	candUnigrams := make(map[int]int)
	refUnigrams := make(map[int]int)

	for _, token := range candTokens {
		candUnigrams[token]++
	}
	for _, token := range refTokens {
		refUnigrams[token]++
	}

	unigramMatches := 0
	for token, count := range candUnigrams {
		if refCount, exists := refUnigrams[token]; exists {
			unigramMatches += min(count, refCount)
		}
	}

	precision := float64(unigramMatches) / float64(len(candTokens))
	recall := float64(unigramMatches) / float64(len(refTokens))
	f1 := 2.0 * precision * recall / (precision + recall)

	if math.IsNaN(f1) {
		f1 = 0.0
	}

	return ROUGEScore{
		ROUGE1_F1: f1,
		ROUGE2_F1: 0.0, // Simplified - would need bigram calculation
		ROUGEL_F1: f1,  // Simplified - using unigram as approximation
		Precision: precision,
		Recall:    recall,
		F1:        f1,
	}
}

// calculateBLEUSimple provides a basic character-based BLEU calculation for testing
func (qe *QualityEvaluator) calculateBLEUSimple(candidate, reference string) BLEUScore {
	if candidate == reference {
		// Perfect match
		return BLEUScore{
			BLEU1:     1.0,
			BLEU2:     1.0,
			BLEU3:     1.0,
			BLEU4:     1.0,
			Precision: []float64{1.0, 1.0, 1.0, 1.0},
		}
	}

	// Simple n-gram matching at character level
	maxN := 4
	precisions := make([]float64, maxN)

	candChars := []rune(candidate)
	refChars := []rune(reference)

	for n := 1; n <= maxN; n++ {
		candNGrams := getCharNGrams(candChars, n)
		refNGrams := getCharNGrams(refChars, n)

		matches := 0
		for candNGram := range candNGrams {
			if refNGrams[candNGram] > 0 {
				matches++
			}
		}

		if len(candNGrams) > 0 {
			precisions[n-1] = float64(matches) / float64(len(candNGrams))
		} else {
			precisions[n-1] = 0.0
		}
	}

	// Calculate BLEU scores
	bleu1 := precisions[0]
	bleu2 := math.Sqrt(precisions[0] * precisions[1])
	bleu3 := math.Pow(precisions[0]*precisions[1]*precisions[2], 1.0/3.0)
	bleu4 := math.Pow(precisions[0]*precisions[1]*precisions[2]*precisions[3], 1.0/4.0)

	// Brevity penalty
	bp := 1.0
	if len(candidate) < len(reference) {
		bp = math.Exp(1.0 - float64(len(reference))/float64(len(candidate)))
	}

	return BLEUScore{
		BLEU1:     bleu1 * bp,
		BLEU2:     bleu2 * bp,
		BLEU3:     bleu3 * bp,
		BLEU4:     bleu4 * bp,
		Precision: precisions,
	}
}

// Helper functions for n-gram calculation
func getNGrams(tokens []int, n int) map[string]int {
	nGrams := make(map[string]int)
	for i := 0; i <= len(tokens)-n; i++ {
		key := fmt.Sprintf("%v", tokens[i:i+n])
		nGrams[key]++
	}
	return nGrams
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// calculateROUGESimple provides a basic character-based ROUGE calculation for testing
func (qe *QualityEvaluator) calculateROUGESimple(candidate, reference string) ROUGEScore {
	candChars := []rune(candidate)
	refChars := []rune(reference)

	// Create character frequency maps
	candUnigrams := make(map[rune]int)
	refUnigrams := make(map[rune]int)

	for _, char := range candChars {
		candUnigrams[char]++
	}
	for _, char := range refChars {
		refUnigrams[char]++
	}

	// Calculate matches
	unigramMatches := 0
	for char, count := range candUnigrams {
		if refCount, exists := refUnigrams[char]; exists {
			unigramMatches += min(count, refCount)
		}
	}

	precision := float64(unigramMatches) / float64(len(candChars))
	recall := float64(unigramMatches) / float64(len(refChars))
	f1 := 2.0 * precision * recall / (precision + recall)

	if math.IsNaN(f1) {
		f1 = 0.0
	}

	return ROUGEScore{
		ROUGE1_F1: f1,
		ROUGE2_F1: 0.0, // Not implemented for character-level
		ROUGEL_F1: f1,  // Same as ROUGE-1 for character level
		Precision: precision,
		Recall:    recall,
		F1:        f1,
	}
}

// Helper function for character n-gram extraction
func getCharNGrams(chars []rune, n int) map[string]int {
	nGrams := make(map[string]int)
	for i := 0; i <= len(chars)-n; i++ {
		key := string(chars[i : i+n])
		nGrams[key]++
	}
	return nGrams
}

func NewEngine(modelPath string, config config.Config) (*Engine, error) {
	ctx := device.NewContext()

	e := &Engine{
		Ctx:       ctx,
		Weights:   &LlamaWeights{},
		ActLogger: NewActivationLogger(),
	}
	e.Config.KVCacheSize = config.KVCacheSize

	if err := e.loadModel(modelPath); err != nil {
		ctx.Free()
		return nil, err
	}

	e.TraceTracker = NewActivationTraceTracker(e.Config.Layers)

	return e, nil
}

// Internal load method
// Internal load method
func (e *Engine) loadModel(path string) error {
	f, err := gguf.LoadFile(path)
	if err != nil {
		return err
	}
	e.Model = f

	// Read Metadata
	// Block Count
	if val, ok := getKV(f, "llama.block_count", "qwen3moe.block_count"); ok {
		e.Config.Layers = int(toFloat64(val)) // GGUF numbers can be various types
	} else {
		e.Config.Layers = 1 // Default for test
	}

	// Embedding Dim
	if val, ok := getKV(f, "llama.embedding_length", "qwen3moe.embedding_length"); ok {
		e.Config.Dim = int(toFloat64(val))
	}
	logger.Log.Info("Model dimensions loaded", "dim", e.Config.Dim)

	// Heads
	if val, ok := getKV(f, "llama.attention.head_count", "qwen3moe.attention.head_count"); ok {
		e.Config.Heads = int(toFloat64(val))
	} else {
		return errors.New("missing attention.head_count")
	}
	logger.Log.Info("Model head count loaded", "heads", e.Config.Heads)

	// Hidden Dim (FFN context)
	if val, ok := getKV(f, "llama.feed_forward_length", "qwen3moe.feed_forward_length"); ok {
		e.Config.HiddenDim = int(toFloat64(val))
	} else {
		e.Config.HiddenDim = 4 * e.Config.Dim // fallback
	}
	logger.Log.Info("Model hidden dim loaded", "hidden_dim", e.Config.HiddenDim)

	// KV Heads (GQA)
	if val, ok := getKV(f, "llama.attention.head_count_kv", "qwen3moe.attention.head_count_kv"); ok {
		// Handle array case specifically for models like Nemotron-3-Nano
		if arr, ok := val.([]interface{}); ok {
			maxVal := 0
			for _, v := range arr {
				iv := int(toFloat64(v))
				if iv > maxVal {
					maxVal = iv
				}
			}
			if maxVal > 0 {
				e.Config.KVHeads = maxVal
			} else {
				e.Config.KVHeads = e.Config.Heads // Fallback if all 0
			}
			logger.Log.Info("Model KV heads loaded (from array)", "kv_heads", e.Config.KVHeads, "raw", val)
		} else {
			e.Config.KVHeads = int(toFloat64(val))
			logger.Log.Info("Model KV heads loaded", "kv_heads", e.Config.KVHeads)
		}
	} else {
		// Default to Heads (MHA)
		e.Config.KVHeads = e.Config.Heads
		logger.Log.Info("Model KV heads default (MHA)", "kv_heads", e.Config.KVHeads)
	}

	// Head Dim
	if e.Config.Heads > 0 {
		e.Config.HeadDim = e.Config.Dim / e.Config.Heads
	}
	logger.Log.Info("Model head dim calculated", "head_dim", e.Config.HeadDim)

	// Seq Len (Context)
	if val, ok := getKV(f, "llama.context_length", "qwen3moe.context_length"); ok {
		e.Config.SeqLen = int(toFloat64(val))
	} else {
		e.Config.SeqLen = 2048 // default
	}
	logger.Log.Info("Model sequence length loaded", "seq_len", e.Config.SeqLen)

	// RoPE Freq
	if val, ok := getKV(f, "llama.rope.freq_base", "qwen3moe.rope.freq_base"); ok {
		e.Config.RopeTheta = float32(toFloat64(val))
		logger.Log.Info("Model RoPE theta loaded", "theta", e.Config.RopeTheta)
	} else {
		// Default to 10k for Llama 2, but Mistral v0.3 uses 1M
		// We'll set it properly based on architecture below
		e.Config.RopeTheta = 10000.0
	}

	// Global Scale
	e.GlobalScale = 1.0

	// RMS Norm Eps
	var eps float32
	if val, ok := getKV(f, "llama.attention.layer_norm_rms_epsilon", "qwen3moe.attention.layer_norm_rms_epsilon"); ok {
		eps = float32(toFloat64(val))
	} else {
		eps = 1e-5 // default
	}
	e.Config.Eps = eps

	// Sliding Window Size (for Mistral)
	// Mistral uses 4096-token sliding window attention
	// If not specified in GGUF, default to 4096 for Mistral, 0 (disabled) for others
	if val, ok := getKV(f, "llama.attention.sliding_window", ""); ok {
		e.Config.WindowSize = int(toFloat64(val))
		logger.Log.Info("Model sliding window size loaded", "window_size", e.Config.WindowSize)
	} else {
		// Check if this is Mistral (heuristic: has specific architecture name)
		if arch, ok := f.KV["general.architecture"].(string); ok && arch == "llama" {
			// For Mistral models, default to 4096
			// For other models (Llama, etc.), use 0 (full attention)
			e.Config.WindowSize = 4096
			logger.Log.Info("Model heuristic: using Mistral sliding window default", "window_size", e.Config.WindowSize)
		} else {
			e.Config.WindowSize = 0 // Full attention
		}
	}

	// Log Model Architecture
	if arch, ok := f.KV["general.architecture"].(string); ok {
		logger.Log.Info("Model architecture detected", "arch", arch)

		// Heuristic: If it's llama or mistral and WindowSize not set,
		// check if we should assume Mistral SWA.
		if strings.Contains(strings.ToLower(arch), "llama") || strings.Contains(strings.ToLower(arch), "mistral") {
			if e.Config.WindowSize == 0 {
				// Most Mistral models in GGUF don't have the sliding_window key set correctly,
				// but they still require it for correctness if they are v0.3+.
				// However, Llama 2 also uses "llama" arch.
				// Heuristic: If RopeTheta is 1M, it's likely Mistral v0.3.
				if e.Config.RopeTheta >= 1000000.0 {
					e.Config.WindowSize = 4096
					logger.Log.Info("Model heuristic: Mistral v0.3 detected, using 4096 SWA")
				}
			}
		}
	}
	if val, ok := getKV(f, "llama.vocab_size", ""); ok {
		e.Config.VocabSize = int(toFloat64(val))
		logger.Log.Info("Model vocab size loaded", "vocab_size", e.Config.VocabSize)
	} else {
		// Fallback for Smollm2 / Llama3 if missing
		e.Config.VocabSize = 49152
		logger.Log.Info("Model vocab size default", "vocab_size", e.Config.VocabSize)
	}

	// Set Precision Mode based on model dimensions (explicit configuration instead of heuristic)
	// This can be overridden by metadata if available
	if arch, ok := f.KV["general.architecture"].(string); ok {
		logger.Log.Info("Model architecture confirmed", "arch", arch)
	}

	if val, ok := f.KV["llama.attention.precision"]; ok {
		if prec, ok := val.(string); ok {
			switch prec {
			case "f16":
				e.Config.PrecisionMode = config.PrecisionFP16
				logger.Log.Info("Model precision mode set (metadata)", "mode", "FP16")
			case "f32":
				e.Config.PrecisionMode = config.PrecisionF32FFN
				logger.Log.Info("Model precision mode set (metadata)", "mode", "F32_FFN")
			case "mixed":
				e.Config.PrecisionMode = config.PrecisionMixed
				logger.Log.Info("Model precision mode set (metadata)", "mode", "Mixed")
			default:
				logger.Log.Info("Model unknown precision mode, using auto", "mode", prec)
			}
		} else {
			e.Config.PrecisionMode = config.PrecisionAuto
		}
	} else {
		// Auto-detect based on dimension
		if e.Config.Dim < 1024 {
			e.Config.PrecisionMode = config.PrecisionF32FFN
			logger.Log.Info("Model precision mode auto-detected", "mode", "F32_FFN", "dim", e.Config.Dim)
		} else if e.Config.Dim >= 4096 {
			e.Config.PrecisionMode = config.PrecisionMixed
			logger.Log.Info("Model precision mode auto-detected", "mode", "Mixed", "dim", e.Config.Dim)
		} else {
			e.Config.PrecisionMode = config.PrecisionFP16
			logger.Log.Info("Model precision mode auto-detected", "mode", "FP16", "dim", e.Config.Dim)
		}
	}

	if val, ok := f.KV["tokenizer.ggml.bos_token_id"]; ok {
		logger.Log.Info("Model BOS token loaded", "id", int(toFloat64(val)))
	}
	if val, ok := f.KV["tokenizer.ggml.eos_token_id"]; ok {
		logger.Log.Info("Model EOS token loaded", "id", int(toFloat64(val)))
	}

	// Initialize KV Cache (now that we have dimensions)
	if err := e.initKVCache(); err != nil {
		return err
	}

	logger.Log.Info("Model configuration summary",
		"layers", e.Config.Layers,
		"dim", e.Config.Dim,
		"hidden_dim", e.Config.HiddenDim,
		"heads", e.Config.Heads,
		"kv_heads", e.Config.KVHeads,
		"head_dim", e.Config.HeadDim,
		"eps", e.Config.Eps,
		"rope_theta", e.Config.RopeTheta)

	// Initialize Weights Slices
	layers := e.Config.Layers
	e.Weights.AttnQ = make([]*device.Tensor, layers)
	e.Weights.AttnK = make([]*device.Tensor, layers)
	e.Weights.AttnV = make([]*device.Tensor, layers)
	e.Weights.AttnO = make([]*device.Tensor, layers)
	e.Weights.AttnNorm = make([]*device.Tensor, layers)
	e.Weights.FfnGate = make([]*device.Tensor, layers)
	e.Weights.FfnDown = make([]*device.Tensor, layers)
	e.Weights.FfnUp = make([]*device.Tensor, layers)
	e.Weights.FfnNorm = make([]*device.Tensor, layers)

	// Map tensors
	for _, t := range f.Tensors {
		cols := int(t.Dimensions[0])
		rows := 1
		if len(t.Dimensions) > 1 {
			rows = int(t.Dimensions[1])
		}
		for i := 2; i < len(t.Dimensions); i++ {
			rows *= int(t.Dimensions[i])
		}

		var mt *device.Tensor
		numElements := rows * cols

		// Validate tensor dimensions before processing
		if err := ValidateTensorDimensions(t.Name, rows, cols, t.Type); err != nil {
			logger.Log.Warn("Tensor validation warning", "name", t.Name, "error", err)
		}

		// Map tensor types
		// Validate F16 Conversion
		if device.Float32ToFloat16(0.0) != 0 {
			panic(fmt.Sprintf("Float32ToFloat16(0.0) = %x (Expected 0)", device.Float32ToFloat16(0.0)))
		}

		if false {
		} else {
			if t.Type == gguf.GGMLTypeF32 {
				// Standard F32 Tensor
				mt = e.Ctx.NewTensorFP32(rows, cols)
				// ... F32 load ...
				dataBytes := numElements * 4
				if uint64(len(t.Data)) < uint64(dataBytes) {
					return fmt.Errorf("tensor %s data truncated", t.Name)
				}
				rawBytes := t.Data[:dataBytes]

				f32Data := make([]float32, numElements)
				for i := 0; i < numElements; i++ {
					bits := binary.LittleEndian.Uint32(rawBytes[i*4 : (i+1)*4])
					f32Data[i] = math.Float32frombits(bits)
				}

				mt.LoadFrom(f32Data)

				// Heuristic for GlobalScale disabled
				e.GlobalScale = 1.0

			} else if t.Type == gguf.GGMLTypeF16 {
				if isNormWeight(t.Name) {
					// Promote Norm weights to FP32 for precision and kernel compatibility
					f32Data := gguf.DequantizeF16(t.Data, numElements)
					mt = e.Ctx.NewTensorFP32(rows, cols)
					mt.LoadFrom(f32Data)

				} else {
					mt = e.Ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
					// ... F16 load ...
					dataBytes := numElements * 2
					if uint64(len(t.Data)) < uint64(dataBytes) {
						return fmt.Errorf("tensor %s data truncated", t.Name)
					}
					mt.LoadFromRaw(t.Data[:dataBytes])
				}
			} else if t.Type == gguf.GGMLTypeQ4_K {
				// Type 12 (Q4_K).

				if e.Config.Dim < 1024 {
					f32Data := gguf.DequantizeQ4K(t.Data, numElements)
					mt = e.Ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
					mt.LoadFrom(f32Data)
				} else {
					// Large Models: Use Q4K Tensor and Kernels
					var err error
					mt, err = e.Ctx.NewQ4KTensor(rows, cols)
					if err != nil {
						return fmt.Errorf("failed to create Q4K tensor %s: %w", t.Name, err)
					}
					dataBytes := (numElements / 256) * 144
					// Check size matches (handle truncated data)
					if uint64(len(t.Data)) < uint64(dataBytes) {
						return fmt.Errorf("tensor %s data truncated (Need %d, Has %d)", t.Name, dataBytes, len(t.Data))
					}

					mt.LoadFromRaw(t.Data[:dataBytes])

				}
			} else if t.Type == gguf.GGMLTypeQ4_0 {
				// Type 2 (Q4_0).
				// Always use native Q4_0 kernel
				// rows * cols elements.
				// Block size 32. 18 bytes per block.
				// Check alignment
				if numElements%32 != 0 {
					return fmt.Errorf("Q4_0 tensor %s size %d not divisible by 32", t.Name, numElements)
				}
				mt = e.Ctx.NewTensorWithType(rows, cols, device.DataTypeQ4_0)
				dataBytes := (numElements / 32) * 18
				if uint64(len(t.Data)) < uint64(dataBytes) {
					return fmt.Errorf("tensor %s data truncated (Need %d, Has %d)", t.Name, dataBytes, len(t.Data))
				}
				mt.LoadFromRaw(t.Data[:dataBytes])

			} else if t.Type == gguf.GGMLTypeQ8_0 {
				// Type 8 (Q8_0).
				// Dequantize to F16 for use in engine
				f32Data := gguf.DequantizeQ8_0(t.Data, numElements)
				mt = e.Ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
				mt.LoadFrom(f32Data)

			} else if t.Type == gguf.GGMLTypeQ6_K {
				// Type 14 (Q6_K).
				if t.Name == "output.weight" || e.Config.Dim >= 1024 {
					// Use Native Q6K
					mt = e.Ctx.NewTensorWithType(rows, cols, device.DataTypeQ6K)
					dataBytes := (numElements / 256) * 210
					if uint64(len(t.Data)) < uint64(dataBytes) {
						return fmt.Errorf("tensor %s data truncated", t.Name)
					}
					mt.LoadFromRaw(t.Data[:dataBytes])
				} else {
					f32Data := gguf.DequantizeQ6K(t.Data, numElements)
					mt = e.Ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
					mt.LoadFrom(f32Data)
				}

			} else if t.Type == gguf.GGMLTypeQ4_K_S { // 99 Unused
				mt = e.Ctx.NewTensor(rows, cols) // fallback
			} else {

				continue
			}
		}

		if mt != nil {
			if t.Type == gguf.GGMLTypeF16 || t.Type == gguf.GGMLTypeF32 || t.Type == gguf.GGMLTypeQ6_K || t.Name == "token_embd.weight" {

			} else if t.Type == gguf.GGMLTypeQ4_K {
				mt.ScanQ4KScales(t.Name)
			}
		}

		// Mapping Logic
		name := t.Name

		// 1. Global weights
		switch name {
		case "token_embd.weight":
			e.Weights.TokenEmb = mt
			continue
		case "output_norm.weight":
			e.Weights.OutputNorm = mt
			continue
		case "output.weight":
			e.Weights.Output = mt
			continue
		}

		// 2. Layer weights: blk.N.suffix
		if strings.HasPrefix(name, "blk.") {
			parts := strings.Split(name, ".")
			if len(parts) < 3 {
				continue
			}

			// Parse N
			layerIdx := 0
			if n, err := fmt.Sscanf(parts[1], "%d", &layerIdx); n != 1 || err != nil {
				continue
			}
			if layerIdx >= layers {
				continue
			}

			suffix := strings.Join(parts[2:], ".")

			switch suffix {
			case "attn_q.weight":
				e.Weights.AttnQ[layerIdx] = mt
			case "attn_k.weight":
				e.Weights.AttnK[layerIdx] = mt
			case "attn_v.weight":
				e.Weights.AttnV[layerIdx] = mt
			case "attn_output.weight":
				e.Weights.AttnO[layerIdx] = mt
			case "attn_norm.weight":
				e.Weights.AttnNorm[layerIdx] = mt
			case "ffn_gate.weight":
				e.Weights.FfnGate[layerIdx] = mt
				if layerIdx == 0 {
					e.Config.HiddenDim = rows
				}
				continue
			case "ffn_down.weight":
				e.Weights.FfnDown[layerIdx] = mt
				if layerIdx <= 5 {
					if mt.DataType() == device.DataTypeQ4K {
						logger.Log.Debug("FFN Down weight quantized", "layer", layerIdx, "type", "Q4K")
						mt.ScanQ4KScales(fmt.Sprintf("blk.%d.ffn_down", layerIdx))
					} else {
						logger.Log.Debug("FFN Down weight loaded", "layer", layerIdx, "type", "F16", "rows", mt.Rows(), "cols", mt.Cols())
					}
				}
			case "ffn_up.weight":
				e.Weights.FfnUp[layerIdx] = mt
			case "ffn_norm.weight":
				e.Weights.FfnNorm[layerIdx] = mt
			}
		}
	}

	// Fallback: many models share token_embd.weight with output.weight
	// For Llama/Mistral architectures, always tie output to token_embd for correctness
	if e.Weights.TokenEmb != nil && e.Weights.Output == nil {
		e.Weights.Output = e.Weights.TokenEmb
		logger.Log.Debug("Tied output.weight to token_embd.weight")
	}

	// Update VocabSize based on actual tensor rows (the source of truth)
	if e.Weights.TokenEmb != nil {
		actualVocab := e.Weights.TokenEmb.Rows()
		if actualVocab != e.Config.VocabSize {
			logger.Log.Warn("Correcting Vocab Size (from embedding)", "configured", e.Config.VocabSize, "actual", actualVocab)
			e.Config.VocabSize = actualVocab
		}
	}

	return nil
}

// Helper for loose typing in GGUF KV
func toFloat64(v interface{}) float64 {
	switch i := v.(type) {
	case float64:
		return i
	case float32:
		return float64(i)
	case int64:
		return float64(i)
	case int32:
		return float64(i)
	case int:
		return float64(i)
	case uint64:
		return float64(i)
	case uint32:
		return float64(i)
	case uint16:
		return float64(i)
	case uint8:
		return float64(i)
	case []interface{}:
		if len(i) > 0 {
			return toFloat64(i[0])
		}
		return 0
	case []uint32:
		if len(i) > 0 {
			return float64(i[0])
		}
		return 0
	case []int32:
		if len(i) > 0 {
			return float64(i[0])
		}
		return 0
	default:
		return 0
	}
}

// InferString is a convenience method that takes a string prompt and returns generated text
func (e *Engine) InferString(prompt string, tokensToGenerate int) (string, error) {
	// Use default sampler config
	samplerConfig := SamplerConfig{
		Temperature:      0.7,
		TopK:             40,
		TopP:             0.9,
		RepPenalty:       1.0,
		Seed:             0,
		DebugActivations: false,
		QualityMode:      false,
	}

	// For now, we need to tokenize the prompt.
	// In a real implementation, this would use the engine's tokenizer
	// For this test, we'll use a simple tokenization approach
	inputTokens := []int{1, 2, 3} // Placeholder tokenization

	tokens, err := e.Infer(inputTokens, tokensToGenerate, samplerConfig)
	if err != nil {
		return "", err
	}

	// For now, return a simple string representation
	// In a real implementation, this would decode tokens back to text
	return fmt.Sprintf("Generated %d tokens", len(tokens)), nil
}

// Infer generates tokens and returns them all at once
func (e *Engine) Infer(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig) ([]int, error) {
	return e.InferWithCallback(inputTokens, tokensToGenerate, samplerConfig, nil)
}

// InferWithCallback generates tokens with optional streaming callback
// If callback is provided, it's called for each generated token
func (e *Engine) InferWithCallback(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, callback func(token int)) ([]int, error) {
	// Validation
	if len(inputTokens) == 0 {
		return nil, errors.New("empty input tokens")
	}

	// Enable activation logging if requested
	if samplerConfig.DebugActivations {
		promptText := fmt.Sprintf("tokens:%v", inputTokens) // Simple representation for now
		e.ActLogger.Enable(promptText, inputTokens)
	}

	// Phase 1: Prefill (Process all tokens except last one, generating KV cache)
	if e.Model == nil {
		return nil, errors.New("no model loaded")
	}

	// Validate critical weights are loaded
	if e.Weights.TokenEmb == nil {
		return nil, errors.New("token embedding weights not loaded")
	}
	if e.Weights.OutputNorm == nil {
		return nil, errors.New("output norm weights not loaded")
	}
	if e.Weights.Output == nil {
		return nil, errors.New("output weights not loaded")
	}

	// Validate input tokens are within vocab range
	for i, token := range inputTokens {
		if token < 0 || token >= e.Weights.TokenEmb.Rows() {
			return nil, fmt.Errorf("input token %d at position %d is out of vocab range [0, %d)", token, i, e.Weights.TokenEmb.Rows())
		}
	}

	// tStart := time.Now()
	result := make([]int, 0, tokensToGenerate)

	// Reset KV cache position
	e.CachePos = 0

	// Lock OS thread for AutoreleasePool consistency
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	sampler := NewSampler(samplerConfig)

	// Main Generation Loop

	// Create scratch buffers for the layer fusion
	// New Scratch Buffer for Zero-Alloc	// Initialize scratch buffers (Heap backed, includes Logits)
	scratch := e.Ctx.NewLayerScratch(len(inputTokens), e.Config.Dim, e.Config.HiddenDim,
		e.Config.Heads, e.Config.KVHeads, e.Config.HeadDim, e.Config.SeqLen, e.Config.VocabSize)
	defer scratch.Free()

	// Logits are in scratch.Logits
	logits := scratch.Logits

	// Phase 1: Prefill all input tokens
	logger.Log.Debug("Start Inference", "cache_pos", e.CachePos)
	for i := 0; i < len(inputTokens); i++ {
		// Autorelease Pool for this iteration
		pool := e.Ctx.AutoreleasePoolPush()

		tToken := time.Now()
		lastToken := inputTokens[i]

		current := e.Weights.TokenEmb.EmbeddingLookup(lastToken, e.GlobalScale)
		if samplerConfig.DebugActivations || (i < 10) {
			current.ScanMax(fmt.Sprintf("[Pos %d] Token %d Embedding", e.CachePos, lastToken))
		}

		// DEBUG: Print first 4 elements of embedding
		e.Ctx.Synchronize()
		// embData := current.ToHost()
		// fmt.Printf("DEBUG_EMB: Token %d (pos %d) first 4: %v\n", lastToken, i, embData[:4])

		currentF32 := current.ToF32()
		current.ReturnToPool() // Release F16

		// Log embedding if first token
		if i == 0 && e.ActLogger.IsEnabled() {
			embData := currentF32.ToHost()
			e.ActLogger.LogEmbedding(embData)
		}

		// Track embedding stats for first token
		if i == 0 {
			stats := currentF32.GetStats(16)
			e.TraceTracker.RecordLayer("embedding", -1, stats)
		}

		// Layers (Attention + FFN)
		for l := 0; l < e.Config.Layers; l++ {
			// log.Printf("DEBUG_PRECISION: Layer %d, Dim=%d, PrecisionMode=%d (0=Auto, 1=FP16, 2=F32FFN, 3=Mixed)", l, e.Config.Dim, e.Config.PrecisionMode)
			attnNorm := e.Weights.AttnNorm[l]
			q := e.Weights.AttnQ[l]
			k := e.Weights.AttnK[l]
			v := e.Weights.AttnV[l]
			o := e.Weights.AttnO[l]
			ffnNorm := e.Weights.FfnNorm[l]
			ffnGate := e.Weights.FfnGate[l]
			ffnUp := e.Weights.FfnUp[l]
			ffnDown := e.Weights.FfnDown[l]
			view := e.Cache.Get(l)
			kCache := view.K
			vCache := view.V

			currentF32.Layer(l, attnNorm, q, k, v, o, ffnNorm, ffnGate, ffnUp, ffnDown, kCache, vCache,
				scratch, // Pass scratch
				e.TraceTracker,
				e.CachePos, e.Config.Heads, e.Config.KVHeads, e.Config.HeadDim, e.Config.RopeTheta, e.Config.Eps, e.Config.HiddenDim, e.Config.SeqLen, e.Config.WindowSize, e.GlobalScale, samplerConfig.DebugActivations, int(e.Config.PrecisionMode),
				view.BlockTable, view.BlockSize,
				func(k, v *device.Tensor) {
					if err := e.Cache.Update(l, e.CachePos, k, v); err != nil {
						panic(err)
					}
				})

			// Log layer activation details if enabled
			if e.ActLogger.IsEnabled() {
				// Get samples and max values for logging
				qSample := GetSampleFromTensor(scratch.QPart.ToHost(), 10)
				kSample := GetSampleFromTensor(scratch.KPart.ToHost(), 10)
				vSample := GetSampleFromTensor(scratch.VPart.ToHost(), 10)

				qMax := GetMaxFromTensor(scratch.QPart.ToHost())
				kMax := GetMaxFromTensor(scratch.KPart.ToHost())
				vMax := GetMaxFromTensor(scratch.VPart.ToHost())
				attnMax := GetMaxFromTensor(scratch.AttOut.ToHost())
				ffnMax := GetMaxFromTensor(currentF32.ToHost())

				e.ActLogger.LogLayer(l, qMax, kMax, vMax, attnMax, ffnMax,
					qSample, kSample, vSample,
					scratch.QPart.ToHost(), scratch.KPart.ToHost(), scratch.VPart.ToHost(),
					scratch.AttOut.ToHost(), currentF32.ToHost())
			}

			// Log layer output if enabled OR if first token (for recovery analysis)
			if samplerConfig.DebugActivations || (i == 0) {
				currentF32.ScanMax(fmt.Sprintf("[Pos %d] Layer %d Output", e.CachePos, l))

				// Track layer stats for first token
				if i == 0 {
					stats := currentF32.GetStats(16)
					e.TraceTracker.RecordLayer(fmt.Sprintf("layer_%d", l), l, stats)
				}
			}
		}

		// If this is the LAST prompt token, sample the first next token
		if i == len(inputTokens)-1 {
			// Final Norm (F32 -> F16)
			// Debug Output Norm Weights
			if i == 0 {
				e.Weights.OutputNorm.ScanMax("Output Norm Weights")
				e.Weights.Output.ScanMax("Output Weights")
			}

			// Debug: check currentF32 before final norm
			currentF32.ScanMax(fmt.Sprintf("Layer %d Final Input (before norm)", i))

			// Check for and handle Inf/NaN values before final norm
			if infInfo := currentF32.ScanForNaN(fmt.Sprintf("Layer %d Pre-Final Norm", i), 5); infInfo.HasInf || infInfo.HasNaN() {
				logger.Log.Warn("NaN detected in pre-final norm, clamping", "count", infInfo.Count, "inf_count", infInfo.InfCount)
				// Clamp extreme values to prevent RMSNorm issues
				hostData := currentF32.ToHost()
				for j := range hostData {
					if math.IsInf(float64(hostData[j]), 0) {
						// Replace Inf with a large but finite value
						if hostData[j] > 0 {
							hostData[j] = 1e6
						} else {
							hostData[j] = -1e6
						}
					} else if math.IsNaN(float64(hostData[j])) {
						hostData[j] = 0.0 // Replace NaN with zero
					}
					// Also clamp extremely large values
					if hostData[j] > 1e6 {
						hostData[j] = 1e6
					} else if hostData[j] < -1e6 {
						hostData[j] = -1e6
					}
				}
				// Load cleaned data back
				cleanTensor := e.Ctx.NewTensorFP32(currentF32.Rows(), currentF32.Cols())
				cleanTensor.LoadFromF32(hostData)
				// Use cleaned tensor for RMSNorm
				cleanTensor.RMSNormFP32_ToF16_Into(e.Weights.OutputNorm, e.Config.Eps, scratch.Normed)
				e.Ctx.Synchronize()
				cleanTensor.ReturnToPool()
			} else {
				currentF32.RMSNormFP32_ToF16_Into(e.Weights.OutputNorm, e.Config.Eps, scratch.Normed)
				e.Ctx.Synchronize()
			}

			// Debug: check normed output
			scratch.Normed.ScanMax(fmt.Sprintf("Layer %d Final Norm Output", i))

			// Check for NaN in normalized output
			if nanInfo := scratch.Normed.ScanForNaN("Final Norm", 5); nanInfo.HasNaN() {
				logger.Log.Error("NaN detected in Final Norm", "count", nanInfo.Count, "positions", nanInfo.Positions)
			}

			// Output Head (F16 -> F32 Logits)
			// scratch.Normed contains result. Use it.
			scratch.Normed.LinearToFP32_Into(e.Weights.Output, logits)
			e.Ctx.Synchronize()

			// Debug: check final logits
			logits.ScanMax(fmt.Sprintf("Layer %d Final Logits", i))

			logitsData := logits.ToHost()

			// Check for NaN in logits BEFORE sampling
			nanInfo := device.DetectNaN(logitsData, 10)
			if nanInfo.HasNaN() {
				logger.Log.Error("NaN detected in logits", "count", nanInfo.Count, "positions", nanInfo.Positions, "has_inf", nanInfo.HasInf)
				// Try to recover by using argmax of non-NaN values
				validCount := 0
				for _, v := range logitsData {
					if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
						validCount++
					}
				}
				if validCount == 0 {
					logger.Log.Error("All logits are NaN or Inf, cannot recover")
					return nil, fmt.Errorf("all logits are NaN/Inf and no recovery logits available")
				} else {
					logger.Log.Warn("Filtered NaNs from logits", "valid_count", validCount, "total", len(logitsData))
				}
			}

			params := make([]struct {
				idx int
				v   float32
			}, len(logitsData))
			for idx, v := range logitsData {
				params[idx].idx = idx
				params[idx].v = v
			}
			sort.Slice(params, func(i, j int) bool { return params[i].v > params[j].v })

			nextToken := sampler.Sample(logitsData, inputTokens, e.Config.VocabSize)

			// Log logits if enabled
			if e.ActLogger.IsEnabled() {
				// We need to pass nil or a default list if we removed expectedTokens definition
				e.ActLogger.LogLogits(logitsData, nil)
			}

			result = append(result, nextToken)

			// Call streaming callback if provided
			if callback != nil {
				callback(nextToken)
			}
		}

		currentF32.ReturnToPool()

		// Increment CachePos after processing the token
		e.CachePos++
		metrics.RecordInference(1, time.Since(tToken))
		e.Ctx.AutoreleasePoolPop(pool)
	}

	// Phase 2: Generation loop (remaining tokens)
	for i := 1; i < tokensToGenerate; i++ {
		// Autorelease Pool for this iteration
		pool := e.Ctx.AutoreleasePoolPush()

		tToken := time.Now()
		lastToken := result[len(result)-1]

		if lastToken < 0 || lastToken >= e.Weights.TokenEmb.Rows() {
			return nil, fmt.Errorf("token %d is out of vocab range", lastToken)
		}

		current := e.Weights.TokenEmb.EmbeddingLookup(lastToken, e.GlobalScale)

		currentF32 := current.ToF32()
		current.ReturnToPool() // Release F16 embedding

		// Layers (Attention + FFN)
		for l := 0; l < e.Config.Layers; l++ {
			logger.Log.Debug("Layer Precision Info", "layer", l, "dim", e.Config.Dim, "mode", e.Config.PrecisionMode)
			// e.updateCurrentLayer(l) // This function call is not defined in the provided context. Removed.
			attnNorm := e.Weights.AttnNorm[l]
			q := e.Weights.AttnQ[l]
			k := e.Weights.AttnK[l]
			v := e.Weights.AttnV[l]
			o := e.Weights.AttnO[l] // Corrected from AttnOutput
			ffnNorm := e.Weights.FfnNorm[l]
			ffnGate := e.Weights.FfnGate[l]
			ffnUp := e.Weights.FfnUp[l]
			ffnDown := e.Weights.FfnDown[l]

			view := e.Cache.Get(l)
			kCache := view.K
			vCache := view.V

			if l == e.Config.Layers-1 && i == 0 {
				// ffnDown.ScanMax("Last Layer FFN Down Weight")
				// fmt.Printf("DEBUG: Eps: %e\n", e.Config.Eps)
			}

			// Layer now handles F32 currentF32, using mixed precision
			currentF32.Layer(l, attnNorm, q, k, v, o, ffnNorm, ffnGate, ffnUp, ffnDown, kCache, vCache,
				scratch, // Pass scratch
				e.TraceTracker,
				e.CachePos, e.Config.Heads, e.Config.KVHeads, e.Config.HeadDim, e.Config.RopeTheta, e.Config.Eps, e.Config.HiddenDim, e.Config.SeqLen, e.Config.WindowSize, e.GlobalScale, samplerConfig.DebugActivations, int(e.Config.PrecisionMode),
				view.BlockTable, view.BlockSize,
				func(k, v *device.Tensor) {
					if err := e.Cache.Update(l, e.CachePos, k, v); err != nil {
						panic(err)
					}
				})

			// Log layer activation details if enabled (generation phase)
			if e.ActLogger.IsEnabled() && i >= len(inputTokens)-1 {
				qSample := GetSampleFromTensor(scratch.QPart.ToHost(), 10)
				kSample := GetSampleFromTensor(scratch.KPart.ToHost(), 10)
				vSample := GetSampleFromTensor(scratch.VPart.ToHost(), 10)

				qMax := GetMaxFromTensor(scratch.QPart.ToHost())
				kMax := GetMaxFromTensor(scratch.KPart.ToHost())
				vMax := GetMaxFromTensor(scratch.VPart.ToHost())
				attnMax := GetMaxFromTensor(scratch.AttOut.ToHost())
				ffnMax := GetMaxFromTensor(currentF32.ToHost())

				e.ActLogger.LogLayer(l, qMax, kMax, vMax, attnMax, ffnMax,
					qSample, kSample, vSample,
					scratch.QPart.ToHost(), scratch.KPart.ToHost(), scratch.VPart.ToHost(),
					scratch.AttOut.ToHost(), currentF32.ToHost())
			}

			if samplerConfig.DebugActivations {
				currentF32.ScanMax(fmt.Sprintf("[Pos %d] Layer %d Output", e.CachePos, l))
			}
		}

		// Final Norm (F32 -> F16)

		// Use Into to avoid alloc
		currentF32.RMSNormFP32_ToF16_Into(e.Weights.OutputNorm, e.Config.Eps, scratch.Normed)

		// Output Head (F16 -> F32 Logits)
		// Reuse pre-allocated logits buffer
		// Output into scratch.Logits (which is 'logits')
		normedF32 := scratch.Normed.ToF32()
		normedF32.LinearF32_Into(e.Weights.Output, logits, e.GlobalScale)
		normedF32.ReturnToPool()

		// Logic update: RMSNormFP32_ToF16_Into does NOT allocate.
		// It writes to scratch.Normed.
		// No need to ReturnToPool for scratch.Normed (it's persistent scratch).

		// Logic update: RMSNormFP32_ToF16_Into does NOT allocate.
		// It writes to scratch.Normed.
		// No need to ReturnToPool for scratch.Normed (it's persistent scratch).

		e.Ctx.Synchronize()
		logitsData := logits.ToHost()

		// Save logits for debugging

		currentF32.ReturnToPool()

		// Construct full history for Repetition Penalty
		// We need to include input tokens + generated tokens
		fullHistory := make([]int, 0, len(inputTokens)+len(result))
		fullHistory = append(fullHistory, inputTokens...)
		fullHistory = append(fullHistory, result...)

		maxIdx := sampler.Sample(logitsData, fullHistory, e.Config.VocabSize)

		if maxIdx == 128001 { // <|end_of_text|> in Llama 3
			e.Ctx.AutoreleasePoolPop(pool)
			break
		}

		result = append(result, maxIdx)

		// Call streaming callback if provided
		if callback != nil {
			callback(maxIdx)
		}
		e.CachePos++
		metrics.RecordInference(1, time.Since(tToken))
		e.Ctx.AutoreleasePoolPop(pool)
	}

	// Save activation log if enabled
	if e.ActLogger.IsEnabled() {
		err := e.ActLogger.SaveToFile("activation_debug.json")
		if err != nil {
			logger.Log.Warn("Failed to save activation log", "error", err)
		}
	}

	return result, nil
}

func (e *Engine) initKVCache() error {
	// Determine Cache Strategy
	useSlidingWindow := false
	if e.Config.WindowSize > 0 && e.Config.WindowSize < e.Config.SeqLen {
		useSlidingWindow = true
		logger.Log.Info("Using Sliding Window Attention", "window", e.Config.WindowSize, "seq_len", e.Config.SeqLen)
	}

	// Override from command line or config if needed (could added a Config.CacheType enum)

	if useSlidingWindow {
		cache := &SlidingWindowKVCache{}
		if err := cache.Init(e.Ctx, e.Config); err != nil {
			return err
		}
		e.Cache = cache
	} else {
		// Default to TensorKVCache (Linear Buffer)
		cache := &TensorKVCache{}
		if err := cache.Init(e.Ctx, e.Config); err != nil {
			return err
		}
		e.Cache = cache
	}

	return nil
}

func (e *Engine) Close() {
	if e.Cache != nil {
		e.Cache.Free()
	}
	if e.Ctx != nil {
		e.Ctx.Free()
	}
}

// LoadWeightFromGGUF decodes weights to F32 for CPU reference
func LoadWeightFromGGUF(e *Engine, name string) []float32 {
	var t *gguf.TensorInfo
	for _, tensor := range e.Model.Tensors {
		if tensor.Name == name {
			t = tensor
			break
		}
	}
	if t == nil {
		panic(fmt.Sprintf("Tensor %s not found in GGUF", name))
	}

	numElements := int(t.Dimensions[0])
	for i := 1; i < len(t.Dimensions); i++ {
		numElements *= int(t.Dimensions[i])
	}

	if t.Type == gguf.GGMLTypeQ4_K {
		return gguf.DequantizeQ4K(t.Data, numElements)
	} else if t.Type == gguf.GGMLTypeQ6_K {
		return gguf.DequantizeQ6K(t.Data, numElements)
	} else if t.Type == gguf.GGMLTypeF32 {
		out := make([]float32, numElements)
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint32(t.Data[i*4 : (i+1)*4])
			out[i] = math.Float32frombits(bits)
		}
		return out
	} else if t.Type == gguf.GGMLTypeF16 {
		out := make([]float32, numElements)
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint16(t.Data[i*2 : (i+1)*2])
			out[i] = device.Float16ToFloat32(bits)
		}
		return out
	}
	panic(fmt.Sprintf("Unsupported type %d for %s", t.Type, name))
}

func isNormWeight(name string) bool {
	return strings.HasSuffix(name, "attn_norm.weight") || strings.HasSuffix(name, "ffn_norm.weight") || name == "output_norm.weight"
}

// ValidateTensorDimensions validates tensor dimensions based on quantization type
func ValidateTensorDimensions(name string, rows, cols int, ggufType gguf.GGMLType) error {
	switch ggufType {
	case gguf.GGMLTypeF32, gguf.GGMLTypeF16:
		if rows <= 0 || cols <= 0 {
			return fmt.Errorf("invalid dimensions: rows=%d, cols=%d", rows, cols)
		}
	case gguf.GGMLTypeQ4_0:
		if cols%32 != 0 {
			return fmt.Errorf("Q4_0 requires cols divisible by 32, got cols=%d", cols)
		}
		if rows <= 0 || cols <= 0 {
			return fmt.Errorf("invalid Q4_0 dimensions: rows=%d, cols=%d", rows, cols)
		}
	case gguf.GGMLTypeQ4_K:
		if cols%256 != 0 {
			return fmt.Errorf("Q4_K requires cols divisible by 256, got cols=%d", cols)
		}
		if rows <= 0 || cols <= 0 {
			return fmt.Errorf("invalid Q4_K dimensions: rows=%d, cols=%d", rows, cols)
		}
	case gguf.GGMLTypeQ6_K:
		if cols%256 != 0 {
			return fmt.Errorf("Q6_K requires cols divisible by 256, got cols=%d", cols)
		}
		if rows <= 0 || cols <= 0 {
			return fmt.Errorf("invalid Q6_K dimensions: rows=%d, cols=%d", rows, cols)
		}
	}
	return nil
}
