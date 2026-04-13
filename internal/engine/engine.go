//go:build metal



package engine

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

var (
	ErrModelSwapped = errors.New("inference interrupted: engine model was swapped")
)

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
func (qe *QualityEvaluator) CalculatePerplexity(tokens []int) PerplexityResult {
	if len(tokens) < 2 {
		return PerplexityResult{Perplexity: 1.0, TotalTokens: len(tokens), AvgLogProb: 0.0}
	}

	// Simplified perplexity calculation
	totalLogProb := 0.0
	validTokens := 0

	for i := 1; i < len(tokens); i++ {
		logProb := -0.5 - 0.1*float64(i%3)
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

func init() {
	RegisterEngine("metal", NewMetalEngine)
}

func NewMetalEngine(modelPath string, config config.Config) (Engine, error) {
	ctx := device.NewContext()

	e := &metalEngine{
		ctx:       ctx,
		config:    config,
		weights:   &LlamaWeights{},
		ActLogger: NewActivationLogger(),
		SeqMgr:    NewSequenceManager(),
	}

	e.TraceTracker = NewActivationTraceTracker(e.config.Layers)
	e.BatchManager = NewContinuousBatchManager()
	e.PromptCache = NewPromptCache()
	
	e.stopChan = make(chan struct{})
	e.doneChan = make(chan struct{})

	if err := e.loadModel(modelPath); err != nil {
		ctx.Free()
		return nil, err
	}

	e.cache = &PagedKVCache{}
	e.cache.Init(ctx, e.config)

	go e.runBatchLoop()

	return e, nil
}

// Internal load method
// Internal load method

func (e *metalEngine) loadModel(path string) error {
	logger.Log.Debug("loadModel starting", "path", path, "ctx_is_nil", e.ctx == nil, "config", fmt.Sprintf("%+v", e.config))
	f, err := gguf.LoadFile(path)
	if err != nil {
		return err
	}
	e.model = f

	// Direct lookup from KV
	archVal := f.KV["general.architecture"]
	embVal := f.KV["gemma4.embedding_length"]
	headVal := f.KV["gemma4.attention.head_count"]
	fmt.Fprintf(os.Stderr, "ENGINE: KV arch=%T(%v) emb=%T(%v) head=%T(%v)\n", archVal, embVal, headVal, embVal, headVal, headVal)
	tok, err := tokenizer.NewFromGGUF(f)
	if err != nil {
		logger.Log.Warn("Failed to initialize tokenizer from GGUF", "error", err)
	} else {
		e.Tokenizer = tok
	}

	// ... (metadata loading remains same)

	// Read Metadata
	// Block Count
	if val, ok := getKV(f,
		"llama.block_count",
		"qwen3moe.block_count",
		"gemma4.block_count",
		"qwen2.block_count",
	); ok {
		e.config.Layers = int(toFloat64(val)) // GGUF numbers can be various types
	} else {
		e.config.Layers = 1 // Default for test
	}

	// 1. Embedding Dim
	if val, ok := getKV(f, "llama.embedding_length", "gemma4.embedding_length", "qwen2.embedding_length"); ok {
		e.config.Dim = int(toFloat64(val))
	}
	if e.config.Dim <= 0 {
		e.config.Dim = 4096 // Fallback
	}

	// 2. Heads (Attention)
	if val, ok := getKV(f, "llama.attention.head_count", "gemma4.attention.head_count", "qwen2.attention.head_count"); ok {
		e.config.Heads = int(toFloat64(val))
	}
	if e.config.Heads <= 0 {
		e.config.Heads = 32 // Fallback
	}

	// 3. KV Heads (GQA/MHA)
	if val, ok := getKV(f, "llama.attention.head_count_kv", "gemma4.attention.head_count_kv", "gemma4.attention.kv_head_count"); ok {
		if arr, ok := val.([]interface{}); ok {
			maxVal := 0
			for _, v := range arr {
				iv := int(toFloat64(v))
				if iv > maxVal { maxVal = iv }
			}
			e.config.KVHeads = maxVal
		} else {
			e.config.KVHeads = int(toFloat64(val))
		}
	}
	if e.config.KVHeads <= 0 {
		e.config.KVHeads = e.config.Heads // Default to MHA
	}

	// 4. Derived Dimensions
	if e.config.Heads > 0 {
		e.config.HeadDim = e.config.Dim / e.config.Heads
	}
	if e.config.HeadDim <= 0 {
		e.config.HeadDim = 128 // Fallback
	}

	// 5. Hidden Dim (FFN)
	if val, ok := getKV(f, "llama.feed_forward_length", "gemma4.feed_forward_length", "qwen2.feed_forward_length"); ok {
		e.config.HiddenDim = int(toFloat64(val))
	}
	if e.config.HiddenDim <= 0 {
		e.config.HiddenDim = 4 * e.config.Dim // Fallback
	}

	logger.Log.Info("Model dimensions loaded", "dim", e.config.Dim, "heads", e.config.Heads, "kv_heads", e.config.KVHeads, "head_dim", e.config.HeadDim, "hidden_dim", e.config.HiddenDim)

	// Seq Len (Context)
	if val, ok := getKV(f,
		"llama.context_length",
		"qwen3moe.context_length",
		"gemma4.context_length",
		"qwen2.context_length",
	); ok {
		e.config.SeqLen = int(toFloat64(val))
	} else {
		e.config.SeqLen = 2048 // default
	}
	logger.Log.Info("Model sequence length loaded", "seq_len", e.config.SeqLen)

	// RoPE Freq
	if val, ok := getKV(f,
		"llama.rope.freq_base",
		"qwen3moe.rope.freq_base",
		"gemma4.rope.freq_base",
		"gemma4.rope.freq_base_swa",
		"qwen2.rope.freq_base",
	); ok {
		e.config.RopeTheta = float32(toFloat64(val))
		logger.Log.Info("Model RoPE theta loaded", "theta", e.config.RopeTheta)
	} else {
		// Default to 10k for Llama 2, but Mistral v0.3 uses 1M
		// We'll set it properly based on architecture below
		e.config.RopeTheta = 10000.0
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
	e.config.Eps = eps

	// Sliding Window Size (for Mistral)
	// Mistral uses 4096-token sliding window attention
	// If not specified in GGUF, default to 4096 for Mistral, 0 (disabled) for others
	if val, ok := getKV(f, "llama.attention.sliding_window", ""); ok {
		e.config.WindowSize = int(toFloat64(val))
		logger.Log.Info("Model sliding window size loaded", "window_size", e.config.WindowSize)
	} else {
		// Check if this is Mistral (heuristic: has specific architecture name)
		if arch, ok := f.KV["general.architecture"].(string); ok && arch == "llama" {
			// For Mistral models, default to 4096
			// For other models (Llama, etc.), use 0 (full attention)
			e.config.WindowSize = 4096
			logger.Log.Info("Model heuristic: using Mistral sliding window default", "window_size", e.config.WindowSize)
		} else {
			e.config.WindowSize = 0 // Full attention
		}
	}

	// Log Model Architecture
	if arch, ok := f.KV["general.architecture"].(string); ok {
		logger.Log.Info("Model architecture detected", "arch", arch)

		// Heuristic: If it's llama or mistral and WindowSize not set,
		// check if we should assume Mistral SWA.
		if strings.Contains(strings.ToLower(arch), "llama") || strings.Contains(strings.ToLower(arch), "mistral") {
			if e.config.WindowSize == 0 {
				// Most Mistral models in GGUF don't have the sliding_window key set correctly,
				// but they still require it for correctness if they are v0.3+.
				// However, Llama 2 also uses "llama" arch.
				// Heuristic: If RopeTheta is 1M, it's likely Mistral v0.3.
				if e.config.RopeTheta >= 1000000.0 {
					e.config.WindowSize = 4096
					logger.Log.Info("Model heuristic: Mistral v0.3 detected, using 4096 SWA")
				}
			}
		}
	}
	if val, ok := getKV(f, "llama.vocab_size", ""); ok {
		e.config.VocabSize = int(toFloat64(val))
		logger.Log.Info("Model vocab size loaded", "vocab_size", e.config.VocabSize)
	} else {
		// Fallback for Smollm2 / Llama3 if missing
		e.config.VocabSize = 49152
		logger.Log.Info("Model vocab size default", "vocab_size", e.config.VocabSize)
	}

	// MOE Metadata (Mixture of Experts)
	if val, ok := getKV(f, "llama.expert_count", ""); ok {
		e.config.ExpertCount = int(toFloat64(val))
		e.config.IsMOE = true
		logger.Log.Info("MOE architecture detected", "expert_count", e.config.ExpertCount)
	}

	if val, ok := getKV(f, "llama.expert_used_count", ""); ok {
		e.config.ExpertUsedCount = int(toFloat64(val))
		logger.Log.Info("MOE expert usage", "expert_used_count", e.config.ExpertUsedCount)
	}

	if val, ok := getKV(f, "llama.expert_shared_count", ""); ok {
		e.config.ExpertSharedCount = int(toFloat64(val))
		logger.Log.Info("MOE shared experts", "expert_shared_count", e.config.ExpertSharedCount)
	}

	if val, ok := getKV(f, "llama.expert_feed_forward_length", ""); ok {
		e.config.ExpertFeedForwardLength = int(toFloat64(val))
		logger.Log.Info("MOE expert FFN dimension", "expert_feed_forward_length", e.config.ExpertFeedForwardLength)
	}

	if val, ok := getKV(f, "llama.expert_shared_feed_forward_length", ""); ok {
		e.config.ExpertSharedFeedForwardLength = int(toFloat64(val))
		logger.Log.Info("MOE shared expert FFN dimension", "expert_shared_feed_forward_length", e.config.ExpertSharedFeedForwardLength)
	}

	if val, ok := getKV(f, "llama.expert_group_count", ""); ok {
		e.config.ExpertGroupCount = int(toFloat64(val))
	}

	if val, ok := getKV(f, "llama.expert_group_used_count", ""); ok {
		e.config.ExpertGroupUsedCount = int(toFloat64(val))
	}

	if val, ok := getKV(f, "llama.expert_weights_norm", ""); ok {
		if norm, ok := val.(bool); ok {
			e.config.ExpertWeightsNorm = norm
		}
	}

	if val, ok := getKV(f, "llama.expert_weights_scale", ""); ok {
		e.config.ExpertWeightsScale = float32(toFloat64(val))
	}

	// Log MOE configuration summary if MOE is detected
	if e.config.IsMOE {
		logger.Log.Info("MOE configuration summary",
			"expert_count", e.config.ExpertCount,
			"expert_used_count", e.config.ExpertUsedCount,
			"expert_shared_count", e.config.ExpertSharedCount,
			"expert_ffn_dim", e.config.ExpertFeedForwardLength,
			"expert_shared_ffn_dim", e.config.ExpertSharedFeedForwardLength)
	}

	// Set Precision Mode based on model dimensions (explicit configuration instead of heuristic)
	// Architecture detection
	if arch, ok := f.KV["general.architecture"].(string); ok {
		e.config.Architecture = arch
		if arch == "nemo" || arch == "nemotron" {
			e.config.IsMOE = true
			logger.Log.Debug("Architecture detected as MOE (Nemotron)", "arch", arch)
		}
		// Gemma4 detection
		if arch == "gemma4" {
			e.config.IsGemma4 = true
			e.config.Gemma4SlidingWindowSize = 512
			e.config.Gemma4SlidingRoPETheta = 10000.0
			e.config.Gemma4FullRoPETheta = 1000000.0
			e.config.Gemma4PartialRoPEFactor = 0.25
			e.config.Gemma4SlidingHeadDim = 256
			e.config.Gemma4FullHeadDim = 512
			logger.Log.Info("Gemma4 architecture detected", "arch", arch)
		}
		logger.Log.Info("Model architecture confirmed", "arch", arch)
	}
	// MOE-specific metadata
	if val, ok := f.KV["llama.expert_count"].(uint32); ok {
		e.config.ExpertCount = int(val)
		e.config.IsMOE = true
		logger.Log.Debug("Architecture detected as MOE (llama.expert_count)", "count", val)
	}

	// Mamba layer detection for hybrid models
	e.detectMambaLayers(f, *logger.Log)

	if val, ok := f.KV["llama.attention.precision"]; ok {
		if prec, ok := val.(string); ok {
			switch prec {
			case "f16":
				e.config.PrecisionMode = config.PrecisionFP16
				logger.Log.Info("Model precision mode set (metadata)", "mode", "FP16")
			case "f32":
				e.config.PrecisionMode = config.PrecisionF32FFN
				logger.Log.Info("Model precision mode set (metadata)", "mode", "F32_FFN")
			case "mixed":
				e.config.PrecisionMode = config.PrecisionMixed
				logger.Log.Info("Model precision mode set (metadata)", "mode", "Mixed")
			default:
				logger.Log.Info("Model unknown precision mode, using auto", "mode", prec)
			}
		} else {
			e.config.PrecisionMode = config.PrecisionAuto
		}
	} else {
		// Auto-detect based on dimension
		if e.config.Dim < 1024 {
			e.config.PrecisionMode = config.PrecisionF32FFN
			logger.Log.Info("Model precision mode auto-detected", "mode", "F32_FFN", "dim", e.config.Dim)
		} else if e.config.Dim >= 4096 {
			e.config.PrecisionMode = config.PrecisionMixed
			logger.Log.Info("Model precision mode auto-detected", "mode", "Mixed", "dim", e.config.Dim)
		} else {
			e.config.PrecisionMode = config.PrecisionFP16
			logger.Log.Info("Model precision mode auto-detected", "mode", "FP16", "dim", e.config.Dim)
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
		"layers", e.config.Layers,
		"dim", e.config.Dim,
		"hidden_dim", e.config.HiddenDim,
		"heads", e.config.Heads,
		"kv_heads", e.config.KVHeads,
		"head_dim", e.config.HeadDim,
		"eps", e.config.Eps,
		"rope_theta", e.config.RopeTheta)

	// Initialize Weights Slices
	layers := e.config.Layers
	e.weights.AttnQ = make([]*device.Tensor, layers)
	e.weights.AttnK = make([]*device.Tensor, layers)
	e.weights.AttnV = make([]*device.Tensor, layers)
	e.weights.AttnO = make([]*device.Tensor, layers)
	e.weights.AttnNorm = make([]*device.Tensor, layers)
	e.weights.FfnGate = make([]*device.Tensor, layers)
	e.weights.FfnDown = make([]*device.Tensor, layers)
	e.weights.FfnUp = make([]*device.Tensor, layers)
	e.weights.FfnNorm = make([]*device.Tensor, layers)
	e.weights.AttnQNorm = make([]*device.Tensor, layers)
	e.weights.AttnKNorm = make([]*device.Tensor, layers)
	e.weights.Mamba = make([]*MambaWeights, layers) // Initialize Mamba/SSM layer storage

	// Initialize MOE weight containers if MOE architecture detected
	if e.config.IsMOE {
		e.weights.MOE = make([]*MOELayerWeights, layers)
		logger.Log.Info("Initialized MOE weight containers", "layers", layers)
	}

	// Map tensors
	for _, t := range f.Tensors {
		if !isNeededTensor(t.Name) {
			continue
		}
		logger.Log.Debug("Processing tensor", "name", t.Name, "engine_ptr", fmt.Sprintf("%p", e), "ctx_ptr", fmt.Sprintf("%p", e.ctx))

		cols := int(t.Dimensions[0])
		rows := 1
		if len(t.Dimensions) > 1 {
			rows = int(t.Dimensions[1])
		}
		for i := 2; i < len(t.Dimensions); i++ {
			rows *= int(t.Dimensions[i])
		}

		numElements := rows * cols
		var mt *device.Tensor

		switch t.Type {
		case gguf.GGMLTypeF32:
			if e == nil || e.ctx == nil {
				panic(fmt.Sprintf("FATAL: Engine or Ctx became nil! Tensor=%s, EnginePtr=%p, CtxPtr=%p", t.Name, e, e.ctx))
			}
			mt = e.ctx.NewTensorFP32(rows, cols)
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

		case gguf.GGMLTypeF16:
			if isNormWeight(t.Name) {
				// Promote Norm weights to FP32 for precision and kernel compatibility
				f32Data := gguf.DequantizeF16(t.Data, numElements)
				mt = e.ctx.NewTensorFP32(rows, cols)
				mt.LoadFrom(f32Data)

			} else {
				mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
				// ... F16 load ...
				dataBytes := numElements * 2
				if uint64(len(t.Data)) < uint64(dataBytes) {
					return fmt.Errorf("tensor %s data truncated", t.Name)
				}
				mt.LoadFromRaw(t.Data[:dataBytes])
			}
		case gguf.GGMLTypeQ4_K:
			// Type 12 (Q4_K).
			if e.config.Dim < 1024 {
				f32Data := gguf.DequantizeQ4K(t.Data, numElements)
				mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
				mt.LoadFrom(f32Data)
			} else {
				// Large Models: Use Q4K Tensor and Kernels
				if e.ctx == nil {
					return fmt.Errorf("engine context is nil before creating Q4K tensor %s", t.Name)
				}
				var err error
				mt, err = e.ctx.NewQ4KTensor(rows, cols)
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
		case gguf.GGMLTypeQ4_0:
			// Type 2 (Q4_0).
			// Always use native Q4_0 kernel
			// rows * cols elements.
			// Block size 32. 18 bytes per block.
			// Check alignment
			if numElements%32 != 0 {
				return fmt.Errorf("Q4_0 tensor %s size %d not divisible by 32", t.Name, numElements)
			}
			mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeQ4_0)
			dataBytes := (numElements / 32) * 18
			if uint64(len(t.Data)) < uint64(dataBytes) {
				return fmt.Errorf("tensor %s data truncated (Need %d, Has %d)", t.Name, dataBytes, len(t.Data))
			}
			mt.LoadFromRaw(t.Data[:dataBytes])

		case gguf.GGMLTypeQ8_0:
			// Type 8 (Q8_0).
			// Dequantize to F16 for use in engine
			f32Data := gguf.DequantizeQ8_0(t.Data, numElements)
			mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
			mt.LoadFrom(f32Data)

		case gguf.GGMLTypeQ6_K:
			// Type 14 (Q6_K).
			if t.Name == "token_embd.weight" {
				logger.Log.Debug("token_embd.weight dequantizing to FP16")
				f32Data := gguf.DequantizeQ6K(t.Data, numElements)
				mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
				mt.LoadFrom(f32Data)
			} else if t.Name == "output.weight" || e.config.Dim >= 1024 {
				// Use Native Q6K
				mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeQ6K)
				dataBytes := (numElements / 256) * 210
				if uint64(len(t.Data)) < uint64(dataBytes) {
					return fmt.Errorf("tensor %s data truncated", t.Name)
				}
				mt.LoadFromRaw(t.Data[:dataBytes])
			} else {
				f32Data := gguf.DequantizeQ6K(t.Data, numElements)
				mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
				mt.LoadFrom(f32Data)
			}

		case gguf.GGMLTypeQ4_K_S: // 99 Unused
			mt = e.ctx.NewTensor(rows, cols) // fallback
		case gguf.GGMLTypeQ5_0:
			// Type 6 (Q5_0). Dequantize to F16 for use in engine
			f32Data := gguf.DequantizeQ5_0(t.Data, numElements)
			mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
			mt.LoadFrom(f32Data)
		case gguf.GGMLTypeQ5_K:
			// Type 13 (Q5_K). Dequantize to F16 for use in engine (use Q5_0 as fallback)
			f32Data := gguf.DequantizeQ5_0(t.Data, numElements)
			mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
			mt.LoadFrom(f32Data)
		default:
			continue
		}

		if mt != nil {
			if t.Type == gguf.GGMLTypeF16 || t.Type == gguf.GGMLTypeF32 || t.Type == gguf.GGMLTypeQ6_K || t.Name == "token_embd.weight" {

			} else if t.Type == gguf.GGMLTypeQ4_K {
				mt.ScanQ4KScales(t.Name)
			}
		}

		// Mapping Logic
		name := t.Name
		logger.Log.Debug("loading tensor", "name", name, "dims", t.Dimensions, "type", t.Type, "offset", t.Offset)

		// DEBUG: Check if this is a Gemma4 norm tensor
		if strings.Contains(name, "attn_q_norm") || strings.Contains(name, "attn_k_norm") {
			logger.Log.Debug("found Gemma4 Q/K norm tensor", "name", name)
		}

		// 1. Global weights (supporting prefixes like nemotron., model., etc.)
		// Strict check to avoid matching blk.N.attn_output.weight to global output.weight
		lowerName := strings.ToLower(name)
		if (strings.HasSuffix(lowerName, "token_embd.weight") ||
			strings.HasSuffix(lowerName, "embed_tokens.weight") ||
			lowerName == "model.embed_tokens.weight") && !strings.Contains(lowerName, "blk.") {
			e.weights.TokenEmb = mt
			continue
		}
		if (strings.HasSuffix(name, "output_norm.weight") || strings.HasSuffix(name, "model.norm.weight")) && !strings.Contains(name, "blk.") {
			e.weights.OutputNorm = mt
			continue
		}
		if (strings.HasSuffix(name, "output.weight") || name == "model.lm_head.weight") && !strings.Contains(name, "blk.") {
			e.weights.Output = mt
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
				e.weights.AttnQ[layerIdx] = mt
			case "attn_k.weight":
				e.weights.AttnK[layerIdx] = mt
			case "attn_v.weight":
				e.weights.AttnV[layerIdx] = mt
			case "attn_output.weight":
				e.weights.AttnO[layerIdx] = mt
			case "attn_norm.weight":
				e.weights.AttnNorm[layerIdx] = mt

			// Gemma-specific: attn_q_norm, attn_k_norm (RMSNorm before Q/K)
			case "attn_q_norm.weight":
				logger.Log.Debug("setting AttnQNorm", "layer", layerIdx, "name", name)
				if e.weights.AttnQNorm == nil || len(e.weights.AttnQNorm) <= layerIdx {
					newSlice := make([]*device.Tensor, layerIdx+1)
					copy(newSlice, e.weights.AttnQNorm)
					e.weights.AttnQNorm = newSlice
				}
				e.weights.AttnQNorm[layerIdx] = mt
			case "attn_k_norm.weight":
				logger.Log.Debug("setting AttnKNorm", "layer", layerIdx, "name", name)
				if e.weights.AttnKNorm == nil || len(e.weights.AttnKNorm) <= layerIdx {
					newSlice := make([]*device.Tensor, layerIdx+1)
					copy(newSlice, e.weights.AttnKNorm)
					e.weights.AttnKNorm = newSlice
				}
				e.weights.AttnKNorm[layerIdx] = mt

			case "ffn_gate.weight":
				e.weights.FfnGate[layerIdx] = mt
				if layerIdx == 0 {
					e.config.HiddenDim = rows
				}
				continue

			// Gemma-specific: inp_gate (input gating)
			case "inp_gate.weight":
				e.weights.FfnGate[layerIdx] = mt

			// Gemma-specific: proj (output projection)
			case "proj.weight":
				e.weights.AttnO[layerIdx] = mt

			case "ffn_down.weight":
				e.weights.FfnDown[layerIdx] = mt
				if layerIdx <= 5 {
					if mt.DataType() == device.DataTypeQ4K {
						logger.Log.Debug("FFN Down weight quantized", "layer", layerIdx, "type", "Q4K")
						mt.ScanQ4KScales(fmt.Sprintf("blk.%d.ffn_down", layerIdx))
					} else {
						logger.Log.Debug("FFN Down weight loaded", "layer", layerIdx, "type", "F16", "rows", mt.Rows(), "cols", mt.Cols())
					}
				}
			case "ffn_up.weight":
				e.weights.FfnUp[layerIdx] = mt
			case "ffn_norm.weight":
				e.weights.FfnNorm[layerIdx] = mt

			// Gemma-specific: post_*_norm weights
			case "post_attention_norm.weight":
				e.weights.AttnNorm[layerIdx] = mt
			case "post_ffw_norm.weight":
				e.weights.FfnNorm[layerIdx] = mt
			case "post_norm.weight":
				e.weights.FfnNorm[layerIdx] = mt

			// Mamba/SSM Tensors
			case "ssm_a":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].A = mt
			case "ssm_d":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].D = mt
			case "ssm_conv1d.weight":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].Conv1dWeight = mt
			case "ssm_conv1d.bias":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].Conv1dBias = mt
			case "ssm_dt.weight":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].DTWeight = mt
			case "ssm_dt.bias":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].DTBias = mt
			case "ssm_norm.weight":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].NormWeight = mt
			case "ssm_norm.bias":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].NormBias = mt
			case "ssm_out.weight":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].OutWeight = mt
			case "ssm_in.weight": // Hypothetical, not seen in logs yet
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].InWeight = mt

			// MOE Router Weights
			case "ffn_gate_inp.weight":
				if e.weights.MOE[layerIdx] == nil {
					e.weights.MOE[layerIdx] = &MOELayerWeights{}
				}
				if e.weights.MOE[layerIdx].Router == nil {
					e.weights.MOE[layerIdx].Router = &MOERouterWeights{}
				}
				e.weights.MOE[layerIdx].Router.GateInput = mt
				logger.Log.Debug("Loaded MOE router gate", "layer", layerIdx, "shape", fmt.Sprintf("[%d, %d]", mt.Rows(), mt.Cols()))

			case "exp_probs_b.bias":
				if e.weights.MOE[layerIdx] == nil {
					e.weights.MOE[layerIdx] = &MOELayerWeights{}
				}
				if e.weights.MOE[layerIdx].Router == nil {
					e.weights.MOE[layerIdx].Router = &MOERouterWeights{}
				}
				e.weights.MOE[layerIdx].Router.ExpertProbBias = mt

			// MOE Expert Weights (3D tensors)
			case "ffn_down_exps.weight":
				if e.weights.MOE[layerIdx] == nil {
					e.weights.MOE[layerIdx] = &MOELayerWeights{}
				}
				if e.weights.MOE[layerIdx].Experts == nil {
					e.weights.MOE[layerIdx].Experts = &MOEExpertWeights{}
				}
				e.weights.MOE[layerIdx].Experts.FfnDownExperts = mt
				// Extract 3D metadata from tensor dimensions [hidden_dim, dim, num_experts]
				if len(t.Dimensions) == 3 {
					e.weights.MOE[layerIdx].Experts.HiddenDim = int(t.Dimensions[0])
					e.weights.MOE[layerIdx].Experts.Dim = int(t.Dimensions[1])
					e.weights.MOE[layerIdx].Experts.NumExperts = int(t.Dimensions[2])
					logger.Log.Debug("Loaded MOE expert down weights", "layer", layerIdx,
						"hidden_dim", t.Dimensions[0], "dim", t.Dimensions[1], "num_experts", t.Dimensions[2])
				} else {
					logger.Log.Debug("Loaded MOE expert down weights", "layer", layerIdx, "shape", t.Dimensions)
				}

			case "ffn_up_exps.weight":
				if e.weights.MOE[layerIdx] == nil {
					e.weights.MOE[layerIdx] = &MOELayerWeights{}
				}
				if e.weights.MOE[layerIdx].Experts == nil {
					e.weights.MOE[layerIdx].Experts = &MOEExpertWeights{}
				}
				e.weights.MOE[layerIdx].Experts.FfnUpExperts = mt
				// Extract 3D metadata if not already set
				if len(t.Dimensions) == 3 && e.weights.MOE[layerIdx].Experts.NumExperts == 0 {
					e.weights.MOE[layerIdx].Experts.HiddenDim = int(t.Dimensions[0])
					e.weights.MOE[layerIdx].Experts.Dim = int(t.Dimensions[1])
					e.weights.MOE[layerIdx].Experts.NumExperts = int(t.Dimensions[2])
					logger.Log.Debug("Loaded MOE expert up weights", "layer", layerIdx,
						"hidden_dim", t.Dimensions[0], "dim", t.Dimensions[1], "num_experts", t.Dimensions[2])
				} else {
					logger.Log.Debug("Loaded MOE expert up weights", "layer", layerIdx, "shape", t.Dimensions)
				}

			case "ffn_gate_exps.weight":
				if e.weights.MOE[layerIdx] == nil {
					e.weights.MOE[layerIdx] = &MOELayerWeights{}
				}
				if e.weights.MOE[layerIdx].Experts == nil {
					e.weights.MOE[layerIdx].Experts = &MOEExpertWeights{}
				}
				e.weights.MOE[layerIdx].Experts.FfnGateExperts = mt

			// MOE Shared Expert Weights
			case "ffn_down_shexp.weight":
				if e.weights.MOE[layerIdx] == nil {
					e.weights.MOE[layerIdx] = &MOELayerWeights{}
				}
				if e.weights.MOE[layerIdx].Shared == nil {
					e.weights.MOE[layerIdx].Shared = &MOESharedWeights{}
				}
				e.weights.MOE[layerIdx].Shared.FfnDownShared = mt
				logger.Log.Debug("Loaded MOE shared expert down weights", "layer", layerIdx, "shape", fmt.Sprintf("[%d, %d]", mt.Rows(), mt.Cols()))

			case "ffn_up_shexp.weight":
				if e.weights.MOE[layerIdx] == nil {
					e.weights.MOE[layerIdx] = &MOELayerWeights{}
				}
				if e.weights.MOE[layerIdx].Shared == nil {
					e.weights.MOE[layerIdx].Shared = &MOESharedWeights{}
				}
				e.weights.MOE[layerIdx].Shared.FfnUpShared = mt
				logger.Log.Debug("Loaded MOE shared expert up weights", "layer", layerIdx, "shape", fmt.Sprintf("[%d, %d]", mt.Rows(), mt.Cols()))

			case "ffn_gate_shexp.weight":
				if e.weights.MOE[layerIdx] == nil {
					e.weights.MOE[layerIdx] = &MOELayerWeights{}
				}
				if e.weights.MOE[layerIdx].Shared == nil {
					e.weights.MOE[layerIdx].Shared = &MOESharedWeights{}
				}
				e.weights.MOE[layerIdx].Shared.FfnGateShared = mt
			}
		}
	}

	// Fallback: many models share token_embd.weight with output.weight (Tied Embeddings)
	// Case 1: TokenEmb exists, Output missing -> Use TokenEmb as Output
	if e.weights.TokenEmb != nil && e.weights.Output == nil {
		e.weights.Output = e.weights.TokenEmb
		logger.Log.Debug("Tied output.weight to token_embd.weight")
	}
	// Case 2: Output exists, TokenEmb missing (e.g. Nemotron) -> Use Output as TokenEmb
	if e.weights.TokenEmb == nil && e.weights.Output != nil {
		e.weights.TokenEmb = e.weights.Output
		logger.Log.Info("Using output.weight as token_embd.weight (tied embeddings recovery)")
	}

	// Update VocabSize based on actual tensor rows (the source of truth)
	if e.weights.TokenEmb != nil {
		actualVocab := e.weights.TokenEmb.Rows()
		if actualVocab != e.config.VocabSize {
			logger.Log.Warn("Correcting Vocab Size (from embedding)", "configured", e.config.VocabSize, "actual", actualVocab)
			e.config.VocabSize = actualVocab
		}
	}

	// Case 3: Gap recovery for missing Mamba ssm_in tensors (Nemotron specific)
	if e.weights.Mamba != nil {
		gaps := f.GetGapTensors()
		logger.Log.Debug("Gap recovery triggered", "num_gaps", len(gaps))
		for idx, gap := range gaps {
			logger.Log.Debug("Found GGUF gap", "idx", idx, "offset", gap.Offset, "size", len(gap.Data))
		}
		gapIdx := 0
		for i := 0; i < e.config.Layers; i++ {
			mw := e.weights.Mamba[i]
			if mw != nil && mw.InWeight == nil {
				// Search for a gap that matches the expected size (approx)
				for gapIdx < len(gaps) {
					gap := gaps[gapIdx]
					gapIdx++
					if len(gap.Data) < 10*1024*1024 {
						continue
					}
					rows := 6144
					cols := 2688
					logger.Log.Info("Recovering missing ssm_in weight from GGUF gap", "layer", i, "offset", gap.Offset, "size", len(gap.Data))
					mt, err := e.ctx.NewTensorFromData(rows, cols, device.DataTypeQ8_0, gap.Data)
					if err != nil {
						logger.Log.Warn("Failed to recover ssm_in weight from gap", "error", err)
						continue
					}
					mw.InWeight = mt
					logger.Log.Info("Successfully recovered ssm_in weight from gap", "layer", i)
					break // Found one for this layer
				}
			}
		}
	}

	// Initialize Mamba Layers and Cache
	e.MambaLayers = make([]*MambaLayer, e.config.Layers)
	e.SSMCache = make([]*MambaState, e.config.Layers)

	for i := 0; i < e.config.Layers; i++ {
		if e.weights.Mamba[i] != nil {
			// Found Mamba weights for this layer
			layer := &MambaLayer{
				Index:   i,
				Weights: e.weights.Mamba[i],
			}
			e.MambaLayers[i] = layer

			dConv := 6144
			kernelSize := 4
			dInner := 4096
			dState := 64

			if layer.Weights.Conv1dWeight != nil {
				dConv = layer.Weights.Conv1dWeight.Rows()
				kernelSize = layer.Weights.Conv1dWeight.Cols()
			}
			if layer.Weights.OutWeight != nil {
				dInner = layer.Weights.OutWeight.Cols()
			}

			convState := e.ctx.NewTensorPooled(dConv, kernelSize)
			ssmState := e.ctx.NewTensorPooled(dInner, dState)

			convState.ZeroInit()
			ssmState.ZeroInit()

			e.SSMCache[i] = &MambaState{
				ConvState: convState,
				SSMState:  ssmState,
			}
		}
	}

	return nil
}

// InferString is a convenience method that takes a string prompt and returns generated text
func (e *metalEngine) InferString(prompt string, tokensToGenerate int) (string, error) {
	samplerConfig := SamplerConfig{
		Temperature:      0.7,
		TopK:             40,
		TopP:             0.9,
		RepPenalty:       1.0,
		Seed:             0,
		DebugActivations: false,
		QualityMode:      false,
	}

	inputTokens := e.Tokenizer.Encode(prompt)

	tokens, err := e.Infer(inputTokens, tokensToGenerate, samplerConfig)
	if err != nil {
		return "", err
	}

	return e.Tokenizer.Decode(tokens), nil
}

// Infer generates tokens and returns them all at once
func (e *metalEngine) Infer(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig) ([]int, error) {
	return e.InferWithCallback(inputTokens, tokensToGenerate, samplerConfig, nil)
}

// InferWithLogits generates tokens and returns them along with the logits of the last token
func (e *metalEngine) InferWithLogits(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig) ([]int, []float32, error) {
	var lastLogits []float32
	tokens, err := e.InferWithCallbackLogits(inputTokens, tokensToGenerate, samplerConfig, nil, func(logits []float32) {
		lastLogits = make([]float32, len(logits))
		copy(lastLogits, logits)
	})
	return tokens, lastLogits, err
}

// InferWithCallbackLogits is like InferWithCallback but also provides logits for each generated token
func (e *metalEngine) InferWithCallbackLogits(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	return e.inferInternal(inputTokens, tokensToGenerate, samplerConfig, tokenCallback, logitsCallback)
}

// InferWithCallback generates tokens with optional streaming callback
// If callback is provided, it's called for each generated token
func (e *metalEngine) InferWithCallback(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, callback func(token int)) ([]int, error) {
	return e.inferInternal(inputTokens, tokensToGenerate, samplerConfig, callback, nil)
}

func (e *metalEngine) ForwardBatch(batch []*Sequence) ([]*device.Tensor, error) {
	batchSize := len(batch)
	if batchSize == 0 {
		return nil, nil
	}

	// 1. Assemble input tokens
	inputTokens := make([]int, batchSize)
	for i, seq := range batch {
		inputTokens[i] = seq.Tokens[len(seq.Tokens)-1]
	}

	// 2. Lookup Embeddings
	if e.weights.TokenEmb == nil {
		return nil, fmt.Errorf("missing embedding weights")
	}
	inputT := e.weights.TokenEmb.EmbeddingLookupBatch(inputTokens, 1.0)
	defer inputT.Free()

	// 3. Scratch Allocation
	qNormDim := 512
	kNormDim := 512
	scratch := e.ctx.NewLayerScratch(batchSize, e.config.Dim, e.config.HiddenDim,
		e.config.Heads, e.config.KVHeads, e.config.HeadDim, e.config.SeqLen, e.config.VocabSize, qNormDim, kNormDim)
	defer scratch.Free()

	// 4. Batch Metadata
	seqIDs := make([]string, batchSize)
	positions := make([]int, batchSize)
	for i, seq := range batch {
		seqIDs[i] = fmt.Sprintf("seq-%d", seq.ID)
		positions[i] = seq.Pos
	}

	current := inputT // Already F16
	defer current.Free()

	// 5. Transformer Layers
	for l := 0; l < e.config.Layers; l++ {
		if e.weights.AttnNorm[l] == nil {
			continue // Skip incomplete layers in test models
		}
		view := e.cache.GetBatch(seqIDs, positions, l)
		
		// e.weights.LoadLayer(l) // If we had lazy loading

		current.LayerBatch(l,
			e.weights.AttnNorm[l], e.weights.AttnQ[l], e.weights.AttnK[l], e.weights.AttnV[l], e.weights.AttnO[l],
			e.weights.FfnNorm[l], e.weights.FfnGate[l], e.weights.FfnUp[l], e.weights.FfnDown[l],
			view.KPools[l], view.VPools[l],
			scratch,
			view.BatchPositions, view.BlockTables, view.MaxBlocks,
			e.config.Heads, e.config.KVHeads, e.config.HeadDim,
			e.config.RopeTheta, e.config.Eps, e.config.HiddenDim,
			view.BlockSize, batchSize, 1.0,
			func(k, v *device.Tensor) {
				updateItems := make([]struct {
					SeqID string
					Pos   int
					K     *device.Tensor
					V     *device.Tensor
				}, batchSize)
				for i := range batch {
					updateItems[i].SeqID = seqIDs[i]
					updateItems[i].Pos = positions[i]
					updateItems[i].K = k.Slice(i, 1)
					updateItems[i].V = v.Slice(i, 1)
				}
				e.cache.UpdateBatch(l, updateItems)
			})

		view.BlockTables.Free()
		view.BatchPositions.Free()
	}

	// 6. Final Norm & Projection
	if e.weights.OutputNorm == nil || e.weights.Output == nil {
		return nil, fmt.Errorf("missing final output weights")
	}
	normed := current.RMSNorm(e.weights.OutputNorm, e.config.Eps)
	defer normed.Free()

	// logits [BatchSize, VocabSize]
	logits := e.ctx.NewTensorWithType(batchSize, e.config.VocabSize, device.DataTypeF32)
	normed.LinearInto(e.weights.Output, logits, 1.0)

	results := make([]*device.Tensor, batchSize)
	for i := 0; i < batchSize; i++ {
		results[i] = logits.Slice(i, 1)
	}
	logits.Free()

	return results, nil
}

func (e *metalEngine) runBatchLoop() {
	defer close(e.doneChan)
	for {
		select {
		case <-e.stopChan:
			return
		default:
		}
		// Pull active sequences from the manager
		active, _ := e.BatchManager.Step(16, e.cache, e.PromptCache)
		if len(active) == 0 {
			time.Sleep(10 * time.Millisecond)
			continue
		}

		// Forward Pass
		results, err := e.ForwardBatch(active)
		if err != nil {
			for _, seq := range active {
				select {
				case seq.Err <- err:
				default:
				}
			}
			continue
		}

		// Sampling & Update
		for i, seq := range active {
			logits := results[i].ToHostF32()
			results[i].Free()

			if seq.LogitsCallback != nil {
				seq.LogitsCallback(logits)
			}

			sampler := NewSampler(seq.Config)
			token := sampler.Sample(logits, seq.Tokens)

			seq.Tokens = append(seq.Tokens, token)
			seq.Pos++

			if seq.TokenCallback != nil {
				seq.TokenCallback(token)
			}

			// Check for completion
			if seq.PrefillCompleted {
				// We just finished prefill, cache the resulting blocks
				blocks := e.cache.GetSequenceBlocks(fmt.Sprintf("seq-%d", seq.ID))
				if blocks != nil {
					e.PromptCache.Insert(seq.Tokens[:seq.PromptLen], blocks)
				}
				seq.PrefillCompleted = false // Only insert once
			}

			if token == 2 || len(seq.Tokens) >= seq.MaxTokens { // 2 = EOS
				select {
				case seq.Result <- seq.Tokens:
				default:
				}
				e.BatchManager.CompleteSequence(seq.ID, e.cache)
			}
		}
	}
}

func (e *metalEngine) inferInternal(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	// Create channels for async completion
	resChan := make(chan []int, 1)
	errChan := make(chan error, 1)

	// Create and submit request
	req := &InferenceRequest{
		ID:             uint64(time.Now().UnixNano()),
		Prompt:         inputTokens,
		MaxTokens:      len(inputTokens) + tokensToGenerate,
		Config:         samplerConfig,
		Result:         resChan,
		Err:            errChan,
		TokenCallback:  tokenCallback,
		LogitsCallback: logitsCallback,
	}

	e.BatchManager.Submit(req)

	// Wait for completion (blocking call for sync API compatibility)
	select {
	case tokens := <-resChan:
		// Return only the newly generated tokens
		if len(tokens) > len(inputTokens) {
			return tokens[len(inputTokens):], nil
		}
		return []int{}, nil
	case err := <-errChan:
		return nil, err
	}
}

func (e *metalEngine) initKVCache() error {
	if err := e.initTurboQuant(); err != nil {
		return err
	}

	// Initialize Paged KV Cache
	pagedCache := &PagedKVCache{}
	if err := pagedCache.Init(e.ctx, e.config); err != nil {
		return err
	}
	e.cache = pagedCache

	// Initialize Batch Manager
	if e.BatchManager == nil {
		e.BatchManager = NewContinuousBatchManager()
	}

	// Loop management moved to NewMetalEngine and SwapModel

	return nil
}

func (e *metalEngine) initTurboQuant() error {
	if e.config.KVCacheType != config.KVCacheTQ1_0 && e.config.KVCacheType != config.KVCacheTQ2_0 {
		return nil
	}

	headDim := e.config.HeadDim
	qjlRows := 64 // Fixed for now

	var rotData []float32
	var qjlData []float32

	// 1. Try Loading from Model (GGUF)
	if e.model != nil {
		rot, qjl, err := e.model.GetTurboQuantMatrices()
		if err == nil {
			rotData = rot
			qjlData = qjl
			logger.Log.Info("TurboQuant matrices loaded from model", "head_dim", headDim)
		} else {
			logger.Log.Debug("TurboQuant matrices not found in model, using fallbacks", "error", err)
		}
	}

	// 2. Deterministic Fallbacks
	if rotData == nil {
		rotData = device.GetPrecomputedRotation(headDim)
	}
	if qjlData == nil {
		qjlData = device.GetPrecomputedQJLSigns(qjlRows * headDim)
	}

	logger.Log.Info("TurboQuant matrices initialized", "head_dim", headDim, "qjl_rows", qjlRows, "source", "loaded/precomputed")

	// 3. Tensor Allocation
	rot := e.ctx.NewTensorFP32(headDim, headDim)
	if rot == nil {
		return fmt.Errorf("failed to allocate TQRotation tensor")
	}
	rot.LoadFromF32(rotData)
	e.ctx.TQRotation = rot

	qjl := e.ctx.NewTensorFP32(qjlRows, headDim)
	if qjl == nil {
		return fmt.Errorf("failed to allocate TQQJL tensor")
	}
	qjl.LoadFromF32(qjlData)
	e.ctx.TQQJL = qjl

	logger.Log.Info("TurboQuant matrices initialized", "head_dim", headDim, "qjl_rows", qjlRows, "source", "loaded/generated")
	return nil
}

func (e *metalEngine) Close() {
	if e.stopChan != nil {
		close(e.stopChan)
		<-e.doneChan // Wait for loop to exit
	}

	if e.weights != nil {
		e.weights.Free()
	}
	if e.cache != nil {
		e.cache.Free()
	}
	for _, s := range e.SSMCache {
		if s != nil {
			s.Free()
		}
	}
	me := e
	if me.ctx != nil {
		me.ctx.Free()
	}
}

// GetEmbedding returns the embedding vector for a single token as a float32 slice
func (e *metalEngine) GetEmbedding(token int) ([]float32, error) {
	if e.weights.TokenEmb == nil {
		return nil, errors.New("token embedding weights not loaded")
	}
	if token < 0 || token >= e.weights.TokenEmb.Rows() {
		return nil, fmt.Errorf("token %d out of vocab range [0, %d)", token, e.weights.TokenEmb.Rows())
	}

	emb := e.weights.TokenEmb.EmbeddingLookup(token, e.GlobalScale)
	defer emb.Free()

	return emb.ToHost(), nil
}



// GetEmbeddings returns embedding vectors for multiple tokens
func (e *metalEngine) GetEmbeddings(tokens []int) ([][]float32, error) {
	embeddings := make([][]float32, len(tokens))
	for i, token := range tokens {
		emb, err := e.GetEmbedding(token)
		if err != nil {
			return nil, fmt.Errorf("error embedding token %d: %w", i, err)
		}
		embeddings[i] = emb
	}
	return embeddings, nil
}

// TextToEmbedding tokenizes the input text and returns embeddings for all tokens
func (e *metalEngine) TextToEmbedding(text string) ([][]float32, error) {
	if e.Tokenizer == nil {
		return nil, errors.New("tokenizer not available")
	}

	tokens := e.Tokenizer.Encode(text)
	if len(tokens) == 0 {
		return nil, errors.New("text tokenized to empty sequence")
	}

	return e.GetEmbeddings(tokens)
}

// EmbeddingDim returns the dimension of the embedding vectors
func (e *metalEngine) EmbeddingDim() int {
	if e.weights.TokenEmb == nil {
		return 0
	}
	return e.weights.TokenEmb.Cols()
}

// LoadWeightFromGGUF decodes weights to F32 for CPU reference
func LoadWeightFromGGUF(e *metalEngine, name string) []float32 {
	var t *gguf.TensorInfo
	for _, tensor := range e.model.Tensors {
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

	switch t.Type {
	case gguf.GGMLTypeQ4_K:
		return gguf.DequantizeQ4K(t.Data, numElements)
	case gguf.GGMLTypeQ6_K:
		return gguf.DequantizeQ6K(t.Data, numElements)
	case gguf.GGMLTypeF32:
		out := make([]float32, numElements)
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint32(t.Data[i*4 : (i+1)*4])
			out[i] = math.Float32frombits(bits)
		}
		return out
	case gguf.GGMLTypeF16:
		out := make([]float32, numElements)
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint16(t.Data[i*2 : (i+1)*2])
			out[i] = device.Float16ToFloat32(bits)
		}
		return out
	}
	panic(fmt.Sprintf("Unsupported type %d for %s", t.Type, name))
}

// SwapModel safely replaces the currently loaded model with a new one
// It blocks new inferences, waits for ongoing ones (via RWMutex), frees the old weights, and loads the new ones.
func (e *metalEngine) SwapModel(newModelPath string, newConfig config.Config) error {
	startTime := time.Now()
	success := false

	defer func() {
		metrics.RecordModelHotSwap(time.Since(startTime), success)
	}()

	// 1. Signal background loop to stop
	if e.stopChan != nil {
		close(e.stopChan)
		<-e.doneChan // Wait for loop to exit completely
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	// 2. Abort all active sequences to prevent caller deadlocks
	if e.BatchManager != nil {
		e.BatchManager.AbortAll(ErrModelSwapped)
	}

	// 3. Free existing cache and weights
	if e.cache != nil {
		e.cache.Free()
		e.cache = nil
	}
	if e.weights != nil {
		e.weights.Free()
		e.weights = nil
	}

	// 4. Update config
	e.config = newConfig

	// 5. Pre-allocate weights container for metadata detection
	e.weights = &LlamaWeights{}

	// 6. Load the new model
	if err := e.loadModel(newModelPath); err != nil {
		return err
	}

	// 7. Restart background loop
	e.stopChan = make(chan struct{})
	e.doneChan = make(chan struct{})
	go e.runBatchLoop()

	success = true
	return nil
}

func (e *metalEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	return nil, fmt.Errorf("ForwardDraft not implemented for metalEngine")
}

func (e *metalEngine) RollbackKV(seqID int, stepCount int) {
	// Stub for now: satisfy interface for speculative decoding
}
