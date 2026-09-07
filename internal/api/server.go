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
}

type TokenizerShim interface {
	Encode(s string) []int
	Decode(tokens []int) string
	GetVocab() []string
}

func (s *Server) getUsedMemory() int64 {
	if s != nil && s.UsedMemory != nil {
		return s.UsedMemory()
	}
	return 0
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

func InitServer(maxMemory int64, memCallback func() int64, e engine.Engine, t TokenizerShim) {
	globalServer = &Server{
		MaxMemory:  maxMemory,
		UsedMemory: memCallback,
		Engine:     e,
		Tokenizer:  t,
	}

	// Start resource monitor background loop
	go globalServer.runResourceMonitor()

	http.HandleFunc("/healthz", globalServer.HealthzEndpoint)
	http.HandleFunc("/v1/completions", globalServer.CompletionsHandler)
	http.HandleFunc("/v1/chat/completions", globalServer.ChatCompletionsHandler)
	http.HandleFunc("/v1/adapters/load", globalServer.LoadAdapterHandler)
	http.HandleFunc("/v1/adapters/list", globalServer.ListAdaptersHandler)
	http.HandleFunc("/readyz", globalServer.ReadyzEndpoint)
}

func (s *Server) runResourceMonitor() {
	ticker := time.NewTicker(1 * time.Second)
	for range ticker.C {
		used := s.getUsedMemory()
		if s.MaxMemory > 0 {
			load := float64(used) / float64(s.MaxMemory)
			if load >= 0.95 {
				// We don't log here to avoid spamming in high load,
				// but metrics are updated in the Healthz endpoint.
			}
		}
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

	// For Phase 3, we pass Speculative and Adapter info via the internal engine logic
	// In a real implementation, we would update the Engine.Infer method or use continuous batching Submit.
	// For now, we'll use the existing sync Infer and assume it handles internal state if possible.

	resTokens, err := s.Engine.Infer(tokens, req.MaxTokens, samplerCfg)
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
	Type     string         `json:"type"`
	Text     string         `json:"text,omitempty"`
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

	for _, msg := range req.Messages {
		promptBuilder.WriteString(fmt.Sprintf("<|im_start|>%s\n", msg.Role))
		switch c := msg.Content.(type) {
		case string:
			promptBuilder.WriteString(c)
		case []interface{}:
			for _, partObj := range c {
				partBytes, _ := json.Marshal(partObj)
				var part ChatContentPart
				if err := json.Unmarshal(partBytes, &part); err == nil {
					if part.Type == "text" {
						promptBuilder.WriteString(part.Text)
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
		}
		promptBuilder.WriteString("<|im_end|>\n")
	}
	promptBuilder.WriteString("<|im_start|>assistant\n")

	// If images were decoded, process through VisionEncoder if available
	if len(imagePayloads) > 0 {
		devCtx := device.NewContext()
		defer devCtx.Free()
		vEncoder := vlm.NewVisionEncoder(devCtx, 512, "clip")
		for _, imgData := range imagePayloads {
			_, _ = vEncoder.Encode(imgData)
		}
	}

	prompt := promptBuilder.String()
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

	resTokens, err := s.Engine.Infer(tokens, maxTokens, samplerCfg)
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
