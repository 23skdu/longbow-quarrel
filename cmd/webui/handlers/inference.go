//go:build webui

package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/23skdu/longbow-quarrel/cmd/webui/config"
	"github.com/23skdu/longbow-quarrel/cmd/webui/engine"
)

// OpenAI-compatible types
type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenAIChatRequest struct {
	Model       string          `json:"model"`
	Messages    []OpenAIMessage `json:"messages"`
	Temperature float64         `json:"temperature,omitempty"`
	TopP        float64         `json:"top_p,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
}

type OpenAIChatChoice struct {
	Index        int           `json:"index"`
	Message      OpenAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type OpenAIChatResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []OpenAIChatChoice `json:"choices"`
	Usage   map[string]int     `json:"usage"`
}

type OpenAICompletionRequest struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	Temperature float64 `json:"temperature,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Stream      bool    `json:"stream,omitempty"`
}

type OpenAICompletionChoice struct {
	Text         string `json:"text"`
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason"`
}

type OpenAICompletionResponse struct {
	ID      string                   `json:"id"`
	Object  string                   `json:"object"`
	Created int64                    `json:"created"`
	Model   string                   `json:"model"`
	Choices []OpenAICompletionChoice `json:"choices"`
	Usage   map[string]int           `json:"usage"`
}

type OpenAIModelsResponse struct {
	Object string        `json:"object"`
	Data   []OpenAIModel `json:"data"`
}

type OpenAIModel struct {
	ID         string `json:"id"`
	Object     string `json:"object"`
	OwnedBy    string `json:"owned_by"`
	Permission []struct {
		ID          string `json:"id"`
		Object      string `json:"object"`
		Created     int64  `json:"created"`
		AllowCreate bool   `json:"allow_create"`
	} `json:"permission"`
}

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
		if err := json.NewEncoder(w).Encode(models); err != nil {
			log.Printf("ERROR: failed to encode models: %v", err)
		}
	}
}

// AdaptersHandler lists the LoRA adapters of the loaded models.
//
// The engine could list them all along: the WebUI had no way to ask, so
// the adapter browser in the UI had nothing to show.
func AdaptersHandler(cfg config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		adapters := engine.ListAdapters()

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(adapters); err != nil {
			log.Printf("ERROR: failed to encode adapters: %v", err)
		}
	}
}

// maxRequestBodyBytes bounds the JSON body a client may send. Prompts are
// small in practice, so anything larger is either a mistake or an attempt to
// exhaust server memory. The limit is applied before decoding so an oversized
// body is never fully read into memory.
const maxRequestBodyBytes = 1 << 20 // 1 MiB

// decodeJSONBody reads a size-limited JSON body into v. It distinguishes an
// oversized body (413) from a malformed one (400) so clients get an
// actionable status code.
//
// encoding/json.Decoder delays read errors and surfaces them as a generic
// *json.SyntaxError, so the *http.MaxBytesError from MaxBytesReader is not
// always recoverable with errors.As. A decode error therefore falls back to
// checking whether ContentLength alone already exceeds the limit, and any
// MaxBytesError still propagating is honoured directly.
// writeJSONError answers with a JSON error document.
//
// The errors used to be written with http.Error, which sends
// text/plain: a client that parses the body as JSON gets a syntax error
// instead of the message.
func writeJSONError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(map[string]string{"error": message}); err != nil {
		log.Printf("ERROR: failed to encode the error response: %v", err)
	}
}

func decodeJSONBody(w http.ResponseWriter, r *http.Request, v any) error {
	limited := http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
	dec := json.NewDecoder(limited)

	// An unknown field is a typo in a parameter name, and a typo that
	// is silently ignored is worse than a rejected request.
	dec.DisallowUnknownFields()

	err := dec.Decode(v)
	if err == nil {
		return nil
	}

	var maxErr *http.MaxBytesError
	tooLarge := errors.As(err, &maxErr) ||
		(r.ContentLength > maxRequestBodyBytes && errors.Is(err, io.EOF)) ||
		r.ContentLength > maxRequestBodyBytes

	if tooLarge {
		writeJSONError(w, http.StatusRequestEntityTooLarge, "Request body too large")
		return err
	}

	writeJSONError(w, http.StatusBadRequest, "Invalid request body")
	return err
}

func GenerateHandler(cfg config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req GenerateRequest
		if err := decodeJSONBody(w, r, &req); err != nil {
			return
		}

		if req.Prompt == "" {
			writeJSONError(w, http.StatusBadRequest, "The prompt is empty")
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
			RecordError("inference_failed")
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
			if resp.Error != "" {
				// The request could not be served (the model failed to
				// load, for example). Report it instead of returning an
				// empty but successful completion.
				log.Printf("Inference failed: %s", resp.Error)
				http.Error(w, resp.Error, http.StatusInternalServerError)
				return
			}
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
		if err := json.NewEncoder(w).Encode(response); err != nil {
			log.Printf("ERROR: failed to encode response: %v", err)
		}
	}
}

func StreamHandler(cfg config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Read and validate the body before committing to SSE headers, so an
		// oversized or malformed request still gets a plain JSON error status
		// rather than a 200 with a text/event-stream body.
		var req GenerateRequest
		if err := decodeJSONBody(w, r, &req); err != nil {
			return
		}

		if req.Prompt == "" {
			writeJSONError(w, http.StatusBadRequest, "The prompt is empty")
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

type HotSwapRequest struct {
	OldModel string `json:"old_model"`
	NewModel string `json:"new_model"`
}

type HotSwapResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message,omitempty"`
}

func HotSwapHandler(cfg config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req HotSwapRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		if req.OldModel == "" || req.NewModel == "" {
			http.Error(w, "old_model and new_model are required", http.StatusBadRequest)
			return
		}

		log.Printf("Hot-swap request: %s -> %s", req.OldModel, req.NewModel)

		adapter := engine.GetAdapter()
		err := adapter.HotSwapModel(req.OldModel, req.NewModel)
		if err != nil {
			log.Printf("Hot-swap error: %v", err)
			http.Error(w, "Hot-swap failed", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(HotSwapResponse{
			Success: true,
			Message: "Model hot-swapped successfully",
		}); err != nil {
			log.Printf("ERROR: failed to encode hot swap response: %v", err)
		}
	}
}

func OpenAIChatCompletionsHandler(cfg config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req OpenAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		if req.MaxTokens <= 0 {
			req.MaxTokens = 256
		}
		if req.Temperature <= 0 {
			req.Temperature = 0.7
		}
		if req.TopP <= 0 {
			req.TopP = 0.95
		}
		if req.Model == "" {
			req.Model = "default"
		}

		var prompt string
		for _, msg := range req.Messages {
			prompt += msg.Role + ": " + msg.Content + "\n"
		}

		adapter := engine.GetAdapter()
		adapterReq := &engine.InferenceRequest{
			Prompt:      prompt,
			Model:       req.Model,
			Temperature: req.Temperature,
			TopP:        req.TopP,
			MaxTokens:   req.MaxTokens,
		}

		responseChanChan, err := adapter.Infer(r.Context(), adapterReq)
		if err != nil {
			http.Error(w, "Inference failed", http.StatusInternalServerError)
			return
		}

		responseChan := <-responseChanChan
		var generatedText string
		tokensGenerated := 0

		for resp := range responseChan {
			tokensGenerated++
			generatedText += resp.Token
			if resp.Complete {
				break
			}
		}

		response := OpenAIChatResponse{
			ID:      "chatcmpl-" + generateID(),
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   req.Model,
			Choices: []OpenAIChatChoice{
				{
					Index: 0,
					Message: OpenAIMessage{
						Role:    "assistant",
						Content: generatedText,
					},
					FinishReason: "stop",
				},
			},
			Usage: map[string]int{
				"prompt_tokens":     len(prompt) / 4,
				"completion_tokens": tokensGenerated,
				"total_tokens":      len(prompt)/4 + tokensGenerated,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			log.Printf("ERROR: failed to encode response: %v", err)
		}
	}
}

func OpenAICompletionsHandler(cfg config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req OpenAICompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		if req.MaxTokens <= 0 {
			req.MaxTokens = 256
		}
		if req.Temperature <= 0 {
			req.Temperature = 0.7
		}
		if req.TopP <= 0 {
			req.TopP = 0.95
		}
		if req.Model == "" {
			req.Model = "default"
		}

		adapter := engine.GetAdapter()
		adapterReq := &engine.InferenceRequest{
			Prompt:      req.Prompt,
			Model:       req.Model,
			Temperature: req.Temperature,
			TopP:        req.TopP,
			MaxTokens:   req.MaxTokens,
		}

		responseChanChan, err := adapter.Infer(r.Context(), adapterReq)
		if err != nil {
			http.Error(w, "Inference failed", http.StatusInternalServerError)
			return
		}

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
		_ = tokensPerSec

		response := OpenAICompletionResponse{
			ID:      "cmpl-" + generateID(),
			Object:  "text_completion",
			Created: time.Now().Unix(),
			Model:   req.Model,
			Choices: []OpenAICompletionChoice{
				{
					Text:         generatedText,
					Index:        0,
					FinishReason: "stop",
				},
			},
			Usage: map[string]int{
				"prompt_tokens":     len(req.Prompt) / 4,
				"completion_tokens": tokensGenerated,
				"total_tokens":      len(req.Prompt)/4 + tokensGenerated,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			log.Printf("ERROR: failed to encode response: %v", err)
		}
	}
}

func OpenAIModelsHandler(cfg config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		adapter := engine.GetAdapter()
		models := adapter.ListModels()

		var modelList []OpenAIModel
		for _, m := range models {
			modelList = append(modelList, OpenAIModel{
				ID:      m.Name,
				Object:  "model",
				OwnedBy: "user",
			})
		}

		response := OpenAIModelsResponse{
			Object: "list",
			Data:   modelList,
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			log.Printf("ERROR: failed to encode response: %v", err)
		}
	}
}

func generateID() string {
	return fmt.Sprintf("%x", time.Now().UnixNano())
}
