package engine

import (
	"context"
	"errors"

	"github.com/23skdu/longbow-quarrel/internal/engine"
)

// ErrQueueFull is returned by an adapter when the request queue is saturated.
//
// The adapters used to return a nil channel and a nil error in that case,
// which a caller cannot distinguish from a successful submission: the request
// simply never completed.
var ErrQueueFull = errors.New("request queue is full")

type InferenceRequest struct {
	Prompt       string
	Temperature  float64
	TopK         int
	TopP         float64
	MaxTokens    int
	Model        string
	Priority     int
	ResponseChan chan chan InferenceResponse
}

type InferenceResponse struct {
	Token    string
	TokenID  int
	Complete bool
	// Error is set when the request could not be served. Without it a
	// failure was indistinguishable from a model that produced no tokens:
	// the client got HTTP 200 and an empty completion.
	Error string
}

type ModelInfo struct {
	Name         string
	Path         string
	Parameters   string
	Quantization string
	Loaded       bool
}

type EngineAdapter interface {
	Infer(ctx context.Context, req *InferenceRequest) (<-chan chan InferenceResponse, error)
	ListModels() []ModelInfo
	LoadModel(modelPath string) error
	UnloadModel(modelPath string)
	HotSwapModel(oldModelPath, newModelPath string) error
	Close()
}

// AdapterInfo describes a LoRA adapter that is loaded.
//
// The engine's own AdapterInfo is the same shape, so the WebUI does not
// need a second type: the adapters report what the engine reports.
type AdapterInfo = engine.AdapterInfo

// AdapterLister is implemented by the adapters that can list the LoRA
// adapters of the loaded models.
//
// It is an optional interface: an adapter that cannot list the adapters
// simply does not implement it, and the browser shows an empty list.
type AdapterLister interface {
	ListAdapters() []AdapterInfo
}

// ListAdapters returns the LoRA adapters of every loaded model, or an
// empty list when the backend cannot report them.
func ListAdapters() []AdapterInfo {
	lister, ok := GetAdapter().(AdapterLister)
	if !ok {
		return []AdapterInfo{}
	}
	adapters := lister.ListAdapters()
	if adapters == nil {
		return []AdapterInfo{}
	}
	return adapters
}

func GetAdapter() EngineAdapter {
	return getAdapterImpl()
}
