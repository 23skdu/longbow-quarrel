//go:build webui

package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"time"

	"github.com/23skdu/longbow-quarrel/cmd/webui/config"
	"github.com/23skdu/longbow-quarrel/cmd/webui/engine"
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

func ModelsHandler(cfg config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		adapter := engine.GetAdapter()
		models := adapter.ListModels()

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

		// Set defaults
		if req.MaxTokens <= 0 {
			req.MaxTokens = 256
		}
		if req.Temperature <= 0 {
			req.Temperature = 0.7
		}
		if req.TopK <= 0 {
			req.TopK = 40
		}
		if req.TopP <= 0 {
			req.TopP = 0.95
		}
		if req.Model == "" {
			req.Model = "default"
		}

		log.Printf("Generate request: prompt=%s, model=%s", req.Prompt, req.Model)

		adapter := engine.GetAdapter()

		// Create inference request
		adapterReq := &engine.InferenceRequest{
			Prompt:      req.Prompt,
			Model:       req.Model,
			Temperature: req.Temperature,
			TopK:        req.TopK,
			TopP:        req.TopP,
			MaxTokens:   req.MaxTokens,
		}

		// Queue inference request
		responseChanChan, err := adapter.Infer(r.Context(), adapterReq)
		if err != nil {
			log.Printf("Inference error: %v", err)
			http.Error(w, "Inference failed", http.StatusInternalServerError)
			return
		}

		if responseChanChan == nil {
			http.Error(w, "Request queue full", http.StatusServiceUnavailable)
			return
		}

		// Collect all tokens
		responseChan := <-responseChanChan
		var generatedText string
		tokensGenerated := 0
		startTime := time.Now()

		for resp := range responseChan {
			tokensGenerated++
			generatedText += resp.Token
			if resp.Complete {
				break
			}
		}

		tokensPerSec := float64(tokensGenerated) / time.Since(startTime).Seconds()

		response := GenerateResponse{
			Text:            generatedText,
			TokensGenerated: tokensGenerated,
			TokensPerSec:    tokensPerSec,
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

		// Set defaults
		if req.MaxTokens <= 0 {
			req.MaxTokens = 256
		}
		if req.Temperature <= 0 {
			req.Temperature = 0.7
		}
		if req.TopK <= 0 {
			req.TopK = 40
		}
		if req.TopP <= 0 {
			req.TopP = 0.95
		}
		if req.Model == "" {
			req.Model = "default"
		}

		log.Printf("Stream request: prompt=%s, model=%s", req.Prompt, req.Model)

		adapter := engine.GetAdapter()

		// Create inference request
		adapterReq := &engine.InferenceRequest{
			Prompt:      req.Prompt,
			Model:       req.Model,
			Temperature: req.Temperature,
			TopK:        req.TopK,
			TopP:        req.TopP,
			MaxTokens:   req.MaxTokens,
		}

		// Queue inference request
		responseChanChan, err := adapter.Infer(r.Context(), adapterReq)
		if err != nil {
			log.Printf("Inference error: %v", err)
			http.Error(w, "Inference failed", http.StatusInternalServerError)
			return
		}

		if responseChanChan == nil {
			http.Error(w, "Request queue full", http.StatusServiceUnavailable)
			return
		}

		// Stream tokens as they arrive
		responseChan := <-responseChanChan
		startTime := time.Now()
		tokensGenerated := 0

		for resp := range responseChan {
			tokensGenerated++
			tokensPerSec := float64(tokensGenerated) / time.Since(startTime).Seconds()

			data := map[string]interface{}{
				"token":          resp.Token,
				"token_id":       resp.TokenID,
				"complete":       resp.Complete,
				"tokens_per_sec": tokensPerSec,
			}

			jsonData, _ := json.Marshal(data)
			w.Write([]byte("data: "))
			w.Write(jsonData)
			w.Write([]byte("\n\n"))
			flusher.Flush()

			if resp.Complete {
				break
			}
		}
	}
}
