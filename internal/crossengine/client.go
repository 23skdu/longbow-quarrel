package crossengine

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type EngineClient interface {
	Generate(prompt string, maxTokens int, temperature float64) (string, error)
	GetTokens() []int
	Name() string
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages   []ChatMessage `json:"messages"`
	MaxTokens  int          `json:"max_tokens,omitempty"`
	Temperature float64     `json:"temperature,omitempty"`
	Stream     bool         `json:"stream,omitempty"`
}

type ChatCompletionChoice struct {
	Message ChatMessage `json:"message"`
	Index   int         `json:"index"`
}

type ChatCompletionResponse struct {
	ID      string                  `json:"id"`
	Choices []ChatCompletionChoice `json:"choices"`
}

type CompletionRequest struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	MaxTokens  int     `json:"max_tokens,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	Stream     bool    `json:"stream,omitempty"`
}

type CompletionChoice struct {
	Text string `json:"text"`
	Index int   `json:"index"`
}

type CompletionResponse struct {
	ID      string             `json:"id"`
	Choices []CompletionChoice `json:"choices"`
}

type VLLMClient struct {
	BaseURL string
	Model  string
	client *http.Client
}

func NewVLLMClient(baseURL, model string) *VLLMClient {
	return &VLLMClient{
		BaseURL: baseURL,
		Model:  model,
		client: &http.Client{Timeout: 60 * time.Second},
	}
}

func (c *VLLMClient) Name() string {
	return "vLLM"
}

func (c *VLLMClient) Generate(prompt string, maxTokens int, temperature float64) (string, error) {
	reqBody := CompletionRequest{
		Model:       c.Model,
		Prompt:      prompt,
		MaxTokens:  maxTokens,
		Temperature: temperature,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	url := c.BaseURL + "/v1/completions"
	resp, err := c.client.Post(url, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return "", fmt.Errorf("vLLM request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("vLLM returned status %d", resp.StatusCode)
	}

	var completionResp CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&completionResp); err != nil {
		return "", err
	}

	if len(completionResp.Choices) == 0 {
		return "", fmt.Errorf("no completion choices returned")
	}

	return completionResp.Choices[0].Text, nil
}

func (c *VLLMClient) GetTokens() []int {
	return []int{}
}

type LlamaCppClient struct {
	BaseURL string
	Model  string
	client *http.Client
}

func NewLlamaCppClient(baseURL, model string) *LlamaCppClient {
	return &LlamaCppClient{
		BaseURL: baseURL,
		Model:  model,
		client: &http.Client{Timeout: 60 * time.Second},
	}
}

func (c *LlamaCppClient) Name() string {
	return "llama.cpp"
}

func (c *LlamaCppClient) Generate(prompt string, maxTokens int, temperature float64) (string, error) {
	url := c.BaseURL + "/v1/completions"

	reqBody := map[string]interface{}{
		"model":       c.Model,
		"prompt":      prompt,
		"max_tokens":  maxTokens,
		"temperature": temperature,
		"stream":      false,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	resp, err := c.client.Post(url, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return "", fmt.Errorf("llama.cpp request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("llama.cpp returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var completionResp CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&completionResp); err != nil {
		return "", err
	}

	if len(completionResp.Choices) == 0 {
		return "", fmt.Errorf("no completion choices returned")
	}

	return completionResp.Choices[0].Text, nil
}

func (c *LlamaCppClient) GetTokens() []int {
	return []int{}
}

type OllamaClient struct {
	BaseURL  string
	Model    string
	client   *http.Client
	lastText string
}

func NewOllamaClient(baseURL, model string) *OllamaClient {
	return &OllamaClient{
		BaseURL: baseURL,
		Model:  model,
		client: &http.Client{Timeout: 120 * time.Second},
	}
}

func (c *OllamaClient) Name() string {
	return "Ollama"
}

func (c *OllamaClient) Generate(prompt string, maxTokens int, temperature float64) (string, error) {
	url := c.BaseURL + "/api/generate"

	reqBody := map[string]interface{}{
		"model":    c.Model,
		"prompt":   prompt,
		"options": map[string]interface{}{
			"num_predict": maxTokens,
			"temperature": temperature,
		},
		"stream": false,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	resp, err := c.client.Post(url, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return "", fmt.Errorf("Ollama request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("Ollama returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var genResp struct {
		Response string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return "", err
	}

	c.lastText = genResp.Response
	return genResp.Response, nil
}

func (c *OllamaClient) GetTokens() []int {
	return Tokenize(c.lastText)
}

func Tokenize(text string) []int {
	var tokens []int
	runes := []rune(text)
	for _, r := range runes {
		tokens = append(tokens, int(r))
	}
	return tokens
}

type QuarrelClient struct {
	BaseURL string
	client  *http.Client
}

func NewQuarrelClient(baseURL string) *QuarrelClient {
	return &QuarrelClient{
		BaseURL: baseURL,
		client: &http.Client{Timeout: 60 * time.Second},
	}
}

func (c *QuarrelClient) Name() string {
	return "Quarrel"
}

func (c *QuarrelClient) Generate(prompt string, maxTokens int, temperature float64) (string, error) {
	url := c.BaseURL + "/v1/completions"

	reqBody := CompletionRequest{
		Model:       "default",
		Prompt:      prompt,
		MaxTokens:  maxTokens,
		Temperature: temperature,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	resp, err := c.client.Post(url, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return "", fmt.Errorf("Quarrel request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("Quarrel returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var completionResp CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&completionResp); err != nil {
		return "", err
	}

	if len(completionResp.Choices) == 0 {
		return "", fmt.Errorf("no completion choices returned")
	}

	return completionResp.Choices[0].Text, nil
}

func (c *QuarrelClient) GetTokens() []int {
	return []int{}
}

func CompareOutputs(a, b string) (float64, int, int) {
	aSet := make(map[string]bool)
	bSet := make(map[string]bool)

	aWords := strings.Fields(a)
	bWords := strings.Fields(b)

	for _, w := range aWords {
		aSet[w] = true
	}
	for _, w := range bWords {
		bSet[w] = true
	}

	intersection := 0
	for w := range aSet {
		if bSet[w] {
			intersection++
		}
	}

	union := len(aSet) + len(bSet) - intersection
	if union == 0 {
		return 0, len(aWords), len(bWords)
	}

	return float64(intersection) / float64(union), len(aWords), len(bWords)
}

func MustParseURL(rawURL string) string {
	rawURL = strings.TrimSpace(rawURL)
	if !strings.HasPrefix(rawURL, "http") {
		rawURL = "http://" + rawURL
	}
	return strings.TrimSuffix(rawURL, "/")
}