package api

import (
	"encoding/json"
	"net/http"
	"time"
)

// Server handles REST endpoints including orchestration health checks.
type Server struct {
	MaxMemory  int64
	UsedMemory func() int64
	Engine     interface {
		Infer(tokens []int, n int, config interface{}) ([]int, error)
	}
	Tokenizer interface {
		Encode(s string) []int
		Decode(tokens []int) string
	}
}

type HealthResponse struct {
	Status      string `json:"status"`
	MemoryUsed  int64  `json:"memory_used"`
	MemoryTotal int64  `json:"memory_total"`
	LoadPercent int    `json:"load_percent"`
}

// HealthzEndpoint enables Kubernetes liveness/readiness orchestration probes.
func (s *Server) HealthzEndpoint(w http.ResponseWriter, r *http.Request) {
	used := s.UsedMemory()
	
	loadPercent := 0
	if s.MaxMemory > 0 {
		loadPercent = int((float64(used) / float64(s.MaxMemory)) * 100)
	}

	resp := HealthResponse{
		MemoryUsed:  used,
		MemoryTotal: s.MaxMemory,
		LoadPercent: loadPercent,
	}

	// Graceful Degradation: Fast fail new requests if OOM is imminent
	if loadPercent >= 95 {
		resp.Status = "degraded (OOM imminent)"
		w.WriteHeader(http.StatusServiceUnavailable) // 503
	} else {
		resp.Status = "ok"
		w.WriteHeader(http.StatusOK) // 200
	}

	if err := json.NewEncoder(w).Encode(resp); err != nil {
		// Response already set or write failed, cannot do much more
	}
}

// Global server instance tracking memory utilization directly from the engine runtime.
var globalServer *Server

func InitServer(maxMemory int64, memCallback func() int64, e interface{}, t interface{}) {
	globalServer = &Server{
		MaxMemory:  maxMemory,
		UsedMemory: memCallback,
		Engine:     e.(EngineShim),
		Tokenizer:  t.(TokenizerShim),
	}
	
	http.HandleFunc("/healthz", globalServer.HealthzEndpoint)
	http.HandleFunc("/v1/completions", globalServer.CompletionsHandler)
}

type EngineShim interface {
	Infer(tokens []int, n int, config interface{}) ([]int, error)
}

type TokenizerShim interface {
	Encode(s string) []int
	Decode(tokens []int) string
}

type CompletionRequest struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
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
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	tokens := s.Tokenizer.Encode(req.Prompt)
	resTokens, err := s.Engine.Infer(tokens, req.MaxTokens, nil)
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
		w.WriteHeader(http.StatusInternalServerError)
	}
}
