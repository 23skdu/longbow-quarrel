//go:build webui

package handlers

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/23skdu/longbow-quarrel/cmd/webui/config"
)

type GenerateRequest struct {
	Prompt      string  `json:"prompt"`
	Model       string  `json:"model,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	TopK        int     `json:"topk,omitempty"`
	TopP        float64 `json:"topp,omitempty"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
}

type GenerateResponse struct {
	Text            string  `json:"text"`
	TokensGenerated int     `json:"tokens_generated"`
	TokensPerSec    float64 `json:"tokens_per_sec"`
}

type ModelInfo struct {
	Name         string `json:"name"`
	Parameters   string `json:"parameters,omitempty"`
	Quantization string `json:"quantization,omitempty"`
	Loaded       bool   `json:"loaded"`
}

func ModelsHandler(cfg config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		models := []ModelInfo{
			{
				Name:         "smollm2",
				Parameters:   "1.7B",
				Quantization: "Q4_K_M",
				Loaded:       true,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(models)
	}
}

func GenerateHandler(cfg config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req GenerateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		log.Printf("Generate request: prompt=%s, model=%s", req.Prompt, req.Model)

		response := GenerateResponse{
			Text:            "This is a placeholder response. The inference engine will be connected in Part 3.",
			TokensGenerated: 10,
			TokensPerSec:    50.0,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}
}

func StreamHandler(cfg config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		var req GenerateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		log.Printf("Stream request: prompt=%s, model=%s", req.Prompt, req.Model)

		for i := 0; i < 10; i++ {
			data := map[string]interface{}{
				"token":    "token",
				"token_id": i,
				"complete": i == 9,
			}

			jsonData, _ := json.Marshal(data)
			w.Write([]byte("data: "))
			w.Write(jsonData)
			w.Write([]byte("\n\n"))
			flusher.Flush()
		}
	}
}
