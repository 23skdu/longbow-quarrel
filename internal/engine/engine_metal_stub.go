//go:build !darwin || !metal

package engine

import (
	"errors"
	"github.com/23skdu/longbow-quarrel/internal/config"
)

// metalEngine is a no-op stub for platforms where Metal is not available.
type PagedKVCache struct{}
type ContinuousBatchManager struct{}
type InferenceRequest struct{}

type metalEngine struct {
	config       config.Config
	cache        *PagedKVCache
	BatchManager *ContinuousBatchManager
	PromptCache  *PromptCache

	// Lifespan Control (Stubbed)
	stopChan chan struct{}
	doneChan chan struct{}
}

func (e *metalEngine) Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error) {
	return nil, errors.New("metal engine not available on this platform")
}

func (e *metalEngine) InferWithLogits(tokens []int, count int, cfg SamplerConfig) ([]int, []float32, error) {
	return nil, nil, errors.New("metal engine not available on this platform")
}

func (e *metalEngine) InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(token int)) ([]int, error) {
	return nil, errors.New("metal engine not available on this platform")
}

func (e *metalEngine) InferWithCallbackLogits(tokens []int, count int, cfg SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	return nil, errors.New("metal engine not available on this platform")
}

func (e *metalEngine) Close() {
	// No-op
}

func (e *metalEngine) SwapModel(modelPath string, cfg config.Config) error {
	return errors.New("metal engine not available on this platform")
}

func (e *metalEngine) Config() config.Config {
	return e.config
}

func (e *metalEngine) GetSeqCachePos(seqID string) int {
	return 0
}

func (e *metalEngine) initKVCache() error {
	return nil
}

func (e *metalEngine) loadModel(path string) error {
	return errors.New("metal engine not available on this platform")
}

func (e *metalEngine) detectMambaLayers(f interface{}, logger interface{}) {
	// No-op
}

func (e *metalEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	return nil, errors.New("metal engine not available on this platform")
}

func (e *metalEngine) RollbackKV(seqID string, newPos int) error {
	return nil
}

func (e *metalEngine) InferString(prompt string, tokensToGenerate int) (string, error) {
	return "", errors.New("metal engine not available on this platform")
}

func (e *metalEngine) inferInternal(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	return nil, errors.New("metal engine not available on this platform")
}

func (e *metalEngine) initTurboQuant() error {
	return errors.New("metal engine not available on this platform")
}

func (e *metalEngine) GetEmbedding(token int) ([]float32, error) {
	return nil, errors.New("metal engine not available on this platform")
}

func (e *metalEngine) GetEmbeddings(tokens []int) ([][]float32, error) {
	return nil, errors.New("metal engine not available on this platform")
}

func (e *metalEngine) TextToEmbedding(text string) ([][]float32, error) {
	return nil, errors.New("metal engine not available on this platform")
}

func (e *metalEngine) EmbeddingDim() int {
	return 0
}

