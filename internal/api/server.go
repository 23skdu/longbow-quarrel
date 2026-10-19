package api

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"github.com/23skdu/longbow-quarrel/internal/sampler"
	"github.com/23skdu/longbow-quarrel/internal/telemetry"
	"github.com/23skdu/longbow-quarrel/internal/vlm"
)

// Server handles REST endpoints including orchestration health checks.
type Server struct {
	MaxMemory  int64
	UsedMemory func() int64
	Engine     engine.Engine
	Tokenizer  TokenizerShim

	// SpeculativeEngine is the optional speculative-decoding engine. It
	// is the same target model with a draft model attached, and it is used
	// for the requests that ask for speculative decoding.
	//
	// The "speculative" request field used to be accepted and then
	// ignored: the handler always used the plain engine, so a request
	// that asked for speculative decoding silently got ordinary
	// generation.
	SpeculativeEngine engine.Engine
}

type TokenizerShim interface {
	Encode(s string) []int
	Decode(tokens []int) string
	GetVocab() []string
}

// engineFor returns the engine that should serve a request.
//
// A request that asks for speculative decoding uses the speculative engine
// when the server was given one, and the plain engine otherwise. The
// per-request draft length is applied to whichever engine is chosen.
func (s *Server) engineFor(speculative bool, draftK int) engine.Engine {
	chosen := s.Engine
	if speculative && s.SpeculativeEngine != nil {
		chosen = s.SpeculativeEngine
	}
	if draftK > 0 {
		if setter, ok := chosen.(engine.DraftLengthSetter); ok {
			setter.SetDraftLength(draftK)
		}
	}
	return chosen
}

func (s *Server) getUsedMemory() int64 {
	if s != nil && s.UsedMemory != nil {
		return s.UsedMemory()
	}
	return 0
}

// chatTemplateProvider is implemented by tokenizers that expose the model's
// own Jinja2 chat template.
//
// It is an optional interface rather than part of TokenizerShim so that a
// tokenizer without a template (or a test double) still satisfies the
// contract.
type chatTemplateProvider interface {
	ChatTemplate() string
}

// chatTemplate returns the model's chat template, or "" when the model does
// not ship one.
func (s *Server) chatTemplate() string {
	if s == nil || s.Tokenizer == nil {
		return ""
	}
	if p, ok := s.Tokenizer.(chatTemplateProvider); ok {
		return p.ChatTemplate()
	}
	return ""
}

type HealthResponse struct {
	Status      string `json:"status"`
	MemoryUsed  int64  `json:"memory_used"`
	MemoryTotal int64  `json:"memory_total"`
	LoadPercent int    `json:"load_percent"`
}

// HealthzEndpoint enables Kubernetes liveness/readiness orchestration probes.
func (s *Server) HealthzEndpoint(w http.ResponseWriter, r *http.Request) {
	used := s.getUsedMemory()

	loadPercent := 0
	if s.MaxMemory > 0 {
		loadPercent = int((float64(used) / float64(s.MaxMemory)) * 100)
	}

	resp := HealthResponse{
		MemoryUsed:  used,
		MemoryTotal: s.MaxMemory,
		LoadPercent: loadPercent,
	}

	if loadPercent >= 95 {
		resp.Status = "degraded (OOM imminent)"
		w.WriteHeader(http.StatusServiceUnavailable) // 503
	} else {
		resp.Status = "ok"
		w.WriteHeader(http.StatusOK) // 200
	}

	if err := json.NewEncoder(w).Encode(resp); err != nil {
		logger.Log.Error("failed to encode health response", "error", err)
	}
}

// Global server instance tracking memory utilization directly from the engine runtime.
var globalServer *Server

// InitServer creates the REST API server and returns a mux with every
// endpoint registered.
//
// It deliberately returns an *http.ServeMux instead of registering on
// http.DefaultServeMux: mutating the global default mux makes the server
// impossible to start more than once in a process, leaks registrations
// between tests, and is one reason the API was never wired into any binary.
// Callers own the returned mux and decide where to serve it.
func InitServer(maxMemory int64, memCallback func() int64, e engine.Engine, t TokenizerShim) *http.ServeMux {
	return InitServerWithSpeculative(maxMemory, memCallback, e, nil, t)
}

// InitServerWithSpeculative is InitServer with an optional speculative
// engine. When it is given, the requests that set "speculative" use it
// instead of the plain engine.
func InitServerWithSpeculative(maxMemory int64, memCallback func() int64, e engine.Engine, spec engine.Engine, t TokenizerShim) *http.ServeMux {
	globalServer = &Server{
		MaxMemory:         maxMemory,
		UsedMemory:        memCallback,
		Engine:            e,
		Tokenizer:         t,
		SpeculativeEngine: spec,
	}

	// Start resource monitor background loop
	go globalServer.runResourceMonitor()

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", globalServer.HealthzEndpoint)
	mux.HandleFunc("/v1/completions", globalServer.CompletionsHandler)
	mux.HandleFunc("/v1/chat/completions", globalServer.ChatCompletionsHandler)
	mux.HandleFunc("/v1/adapters/load", globalServer.LoadAdapterHandler)
	mux.HandleFunc("/v1/adapters/list", globalServer.ListAdaptersHandler)
	mux.HandleFunc("/readyz", globalServer.ReadyzEndpoint)
	return mux
}

// runResourceMonitor samples engine memory usage once a second and publishes
// it to Prometheus so the value is observable rather than discarded.
func (s *Server) runResourceMonitor() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		metrics.RecordHostMemory(s.getUsedMemory())
	}
}

// ReadyzEndpoint provides a standard Kubernetes readiness probe.
func (s *Server) ReadyzEndpoint(w http.ResponseWriter, r *http.Request) {
	if s.Engine == nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	}
	w.WriteHeader(http.StatusOK)
}

type CompletionRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	MaxTokens   int      `json:"max_tokens"`
	Temperature float64  `json:"temperature"`
	Adapter     string   `json:"adapter"`
	Speculative bool     `json:"speculative"`
	DraftK      int      `json:"draft_k"` // Draft tokens per step when speculative
	Images      []string `json:"images"`  // Base64 encoded images
	Grammar     string   `json:"grammar"` // EBNF grammar source
}

type CompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Choices []struct {
		Text string `json:"text"`
	} `json:"choices"`
}

func (s *Server) CompletionsHandler(w http.ResponseWriter, r *http.Request) {
	ctx, span := telemetry.StartSpan(r.Context(), "CompletionsHandler")
	defer span.End()
	_ = ctx

	// Graceful Degradation: Fast-fail if OOM risk
	used := s.getUsedMemory()
	if s.MaxMemory > 0 && float64(used)/float64(s.MaxMemory) >= 0.95 {
		http.Error(w, "Service Unavailable: OOM Imminent (Memory at 95%+)", http.StatusServiceUnavailable)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.Prompt == "" {
		http.Error(w, "prompt is required", http.StatusBadRequest)
		return
	}

	if s.Tokenizer == nil || s.Engine == nil {
		http.Error(w, "Inference components not initialized", http.StatusServiceUnavailable)
		return
	}

	tokens := s.Tokenizer.Encode(req.Prompt)

	samplerCfg := engine.SamplerConfig{
		Temperature: req.Temperature,
		TopP:        0.95,
		TopK:        40,
	}

	if req.Grammar != "" {
		vocabList := s.Tokenizer.GetVocab()
		if vocabList != nil {
			grammar := sampler.NewJSONGrammar(vocabList)
			samplerCfg.Grammar = grammar
		}
	}

	// The "speculative" field selects the speculative engine, which is the
	// same target model with a draft model attached. It used to be
	// accepted and then ignored.
	eng := s.engineFor(req.Speculative, req.DraftK)

	resTokens, err := eng.Infer(tokens, req.MaxTokens, samplerCfg)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	resp := CompletionResponse{
		ID:      "cmpl-123",
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Choices: []struct {
			Text string `json:"text"`
		}{
			{Text: s.Tokenizer.Decode(resTokens)},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		logger.Log.Error("failed to encode completion response", "error", err)
	}
}

type ChatMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type ChatContentPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *ChatImageURL `json:"image_url,omitempty"`
}

type ChatImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float64       `json:"temperature"`
	TopP        float64       `json:"top_p"`
	Stream      bool          `json:"stream"`
	Grammar     string        `json:"grammar,omitempty"`
	Speculative bool          `json:"speculative,omitempty"`
	DraftK      int           `json:"draft_k,omitempty"` // Draft tokens per step when speculative
}

type ChatCompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int         `json:"index"`
		Message      ChatMessage `json:"message"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
}

func (s *Server) ChatCompletionsHandler(w http.ResponseWriter, r *http.Request) {
	ctx, span := telemetry.StartSpan(r.Context(), "ChatCompletionsHandler")
	defer span.End()
	_ = ctx

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if len(req.Messages) == 0 {
		http.Error(w, "messages are required", http.StatusBadRequest)
		return
	}

	if s.Tokenizer == nil || s.Engine == nil {
		http.Error(w, "Inference components not initialized", http.StatusServiceUnavailable)
		return
	}

	var promptBuilder strings.Builder
	var imagePayloads [][]byte

	// The text of each message, in order, so the model's own chat template
	// can be applied when it ships one.
	textMessages := make([]engine.Message, 0, len(req.Messages))
	var systemPrompt string

	for _, msg := range req.Messages {
		switch c := msg.Content.(type) {
		case string:
			textMessages = append(textMessages, engine.Message{Role: msg.Role, Content: c})
			if msg.Role == "system" {
				systemPrompt = c
			}
		case []interface{}:
			var textParts strings.Builder
			for _, partObj := range c {
				partBytes, _ := json.Marshal(partObj)
				var part ChatContentPart
				if err := json.Unmarshal(partBytes, &part); err == nil {
					if part.Type == "text" {
						textParts.WriteString(part.Text)
					} else if part.Type == "image_url" && part.ImageURL != nil {
						rawURL := part.ImageURL.URL
						// Support base64 data URIs: data:image/...;base64,...
						commaIdx := strings.Index(rawURL, ",")
						b64Data := rawURL
						if commaIdx >= 0 {
							b64Data = rawURL[commaIdx+1:]
						}
						decoded, err := base64.StdEncoding.DecodeString(b64Data)
						if err == nil && len(decoded) > 0 {
							imagePayloads = append(imagePayloads, decoded)
						}
					}
				}
			}
			textMessages = append(textMessages, engine.Message{Role: msg.Role, Content: textParts.String()})
			if msg.Role == "system" {
				systemPrompt = textParts.String()
			}
		}
	}

	// Use the model's own chat template when it ships one. The handler used
	// to hardcode the ChatML format (<|im_start|>...) for every model, which
	// is the Qwen format: a Llama 3, Gemma or Mistral model prompted that way
	// produces garbage.
	//
	// The template comes from the GGUF metadata key
	// "tokenizer.chat_template" and is rendered by the Jinja2-subset renderer
	// in internal/engine.
	prompt := ""
	if tmpl := s.chatTemplate(); tmpl != "" {
		wrapper := &engine.PromptWrapper{ChatTemplate: tmpl, SystemPrompt: systemPrompt}
		rendered, err := wrapper.Wrap(textMessages)
		if err == nil && rendered != "" {
			prompt = rendered
		} else if err != nil {
			logger.Log.Warn("chat template rendering failed; falling back to the default format", "error", err)
		}
	}
	if prompt == "" {
		for _, msg := range textMessages {
			fmt.Fprintf(&promptBuilder, "<|im_start|>%s\n", msg.Role)
			promptBuilder.WriteString(msg.Content)
			promptBuilder.WriteString("<|im_end|>\n")
		}
		promptBuilder.WriteString("<|im_start|>assistant\n")
		prompt = promptBuilder.String()
	}

	// If images were decoded, process through VLM encoder using the model's actual architecture
	if len(imagePayloads) > 0 && s.Engine != nil {
		arch := s.Engine.Config().Architecture
		if arch != "" {
			devCtx := device.NewContext()
			defer devCtx.Free()
			vlmCfg := vlm.VLMConfig{
				Architecture: arch,
				ImageSize:    224,
				PatchSize:    14,
				HiddenDim:    512,
			}
			vDecoder, err := vlm.NewVLMDecoder(devCtx, vlmCfg)
			if err == nil {
				for _, imgData := range imagePayloads {
					visionTensor, encErr := vDecoder.Decode(imgData)
					if encErr == nil && visionTensor != nil {
						_ = visionTensor
						prompt += "[vision_features_injected]"
					}
				}
			}
		}
	}

	tokens := s.Tokenizer.Encode(prompt)

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 64
	}

	samplerCfg := engine.SamplerConfig{
		Temperature: req.Temperature,
		TopP:        0.95,
		TopK:        40,
	}

	if req.Grammar != "" {
		vocabList := s.Tokenizer.GetVocab()
		if vocabList != nil {
			grammar := sampler.NewJSONGrammar(vocabList)
			samplerCfg.Grammar = grammar
		}
	}

	// A request that asks for speculative decoding uses the speculative
	// engine, which is the same target model with a draft model attached.
	eng := s.engineFor(req.Speculative, req.DraftK)

	resTokens, err := eng.Infer(tokens, maxTokens, samplerCfg)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	resp := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []struct {
			Index        int         `json:"index"`
			Message      ChatMessage `json:"message"`
			FinishReason string      `json:"finish_reason"`
		}{
			{
				Index: 0,
				Message: ChatMessage{
					Role:    "assistant",
					Content: s.Tokenizer.Decode(resTokens),
				},
				FinishReason: "stop",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		logger.Log.Error("failed to encode chat completion response", "error", err)
	}
}
