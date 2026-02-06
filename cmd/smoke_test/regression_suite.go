//go:build darwin && metal

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"time"
	"unicode"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

type BaselineEntry struct {
	Prompt     string    `json:"prompt"`
	Tokens     []int     `json:"tokens"`
	Logits     []float32 `json:"logits"`
	Perplexity float64   `json:"perplexity"`
}

type BaselineFile struct {
	ModelName string                   `json:"model_name"`
	Timestamp string                   `json:"timestamp"`
	Entries   map[string]BaselineEntry `json:"entries"`
}

func calculatePerplexity(logits []float32, targetToken int) float64 {
	var maxLogit float32 = -math.MaxFloat32
	for _, l := range logits {
		if l > maxLogit {
			maxLogit = l
		}
	}

	var sumExp float64
	for _, l := range logits {
		sumExp += math.Exp(float64(l - maxLogit))
	}

	logSumExp := float64(maxLogit) + math.Log(sumExp)
	logProb := float64(logits[targetToken]) - logSumExp
	return -logProb
}

func calculateMSE(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.MaxFloat64
	}
	var sum float64
	for i := range a {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}
	return sum / float64(len(a))
}

func isCoherent(text string) bool {
	if text == "" {
		return false
	}

	if len(text) < 2 {
		return true
	}

	consecutiveCount := 0
	prevChar := rune(0)
	for i, c := range text {
		if i > 0 && c == prevChar {
			consecutiveCount++
			if consecutiveCount > 3 {
				return false
			}
		} else {
			consecutiveCount = 1
		}
		prevChar = c
	}

	hasLatin := false
	hasCyrillic := false
	hasControl := false
	for _, c := range text {
		if unicode.IsLetter(c) || unicode.IsNumber(c) {
			if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') {
				hasLatin = true
			}
			if (c >= 'а' && c <= 'я') || (c >= 'А' && c <= 'Я') {
				hasCyrillic = true
			}
		}
		if unicode.IsControl(c) && c != '\n' && c != '\t' {
			hasControl = true
		}
	}

	if hasLatin && hasCyrillic {
		return false
	}

	if hasControl {
		return false
	}

	punctRun := 0
	for _, c := range text {
		if c == '!' || c == '?' || c == '.' {
			punctRun++
			if punctRun > 5 {
				return false
			}
		} else {
			punctRun = 0
		}
	}

	return true
}

type MemoryManager struct {
	maxMemoryBytes   int64
	cleanupThreshold float64
}

func NewMemoryManager() *MemoryManager {
	return &MemoryManager{
		maxMemoryBytes:   16 * 1024 * 1024 * 1024,
		cleanupThreshold: 0.8,
	}
}

func (mm *MemoryManager) CheckMemory() bool {
	alloced := device.AllocatedBytes()
	if alloced > int64(float64(mm.maxMemoryBytes)*mm.cleanupThreshold) {
		fmt.Printf("Memory usage high: %d bytes (%.1f%%), triggering cleanup\n",
			alloced, float64(alloced)/float64(mm.maxMemoryBytes)*100)
		return mm.forceCleanup()
	}
	return true
}

func (mm *MemoryManager) forceCleanup() bool {
	fmt.Println("Forcing garbage collection and memory cleanup...")
	runtime.GC()
	runtime.KeepAlive(nil)

	time.Sleep(100 * time.Millisecond)

	alloced := device.AllocatedBytes()
	fmt.Printf("Memory after cleanup: %d bytes\n", alloced)
	return alloced < int64(float64(mm.maxMemoryBytes)*mm.cleanupThreshold)
}

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model")
	baselinePath := flag.String("baseline", "", "Path to baseline JSON file")
	generate := flag.Bool("generate", false, "Generate baseline instead of checking")
	threshold := flag.Float64("threshold", 1e-5, "MSE threshold for failure")
	pplThreshold := flag.Float64("ppl-threshold", 0.05, "Perplexity difference threshold")
	maxMemory := flag.Int64("max-memory", 16*1024*1024*1024, "Maximum memory in bytes")
	flag.Parse()

	memMgr := NewMemoryManager()
	memMgr.maxMemoryBytes = *maxMemory
	device.MaxGPUMemory = *maxMemory

	success := true
	if *modelPath == "" {
		fmt.Println("Running default regression suite for key models...")
		models := []struct {
			name string
			path string
		}{
			{"gpt-oss", "/Users/rsd/.ollama/models/blobs/sha256-e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb"},
			{"mistral", "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"},
			{"granite4", "/Users/rsd/.ollama/models/blobs/sha256-5c7ac4aead1bcf4c8da9534ed72cc632d005aeed6547f1e8662ccdfae688364e"},
			{"tinyllama", "/Users/rsd/.ollama/models/blobs/sha256-2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816"},
		}

		for i, m := range models {
			fmt.Printf("\n=== Testing Model: %s (%d/%d) ===\n", m.name, i+1, len(models))
			if _, err := os.Stat(m.path); os.IsNotExist(err) {
				fmt.Printf("Skipping %s: file not found at %s\n", m.name, m.path)
				continue
			}

			if !memMgr.CheckMemory() {
				fmt.Printf("Memory check failed before loading %s, stopping\n", m.name)
				success = false
				break
			}

			if !runModelTest(m.path, *baselinePath, *generate, *threshold, *pplThreshold, memMgr) {
				success = false
			}

			if !memMgr.forceCleanup() {
				fmt.Printf("Failed to cleanup memory after %s, stopping\n", m.name)
				success = false
				break
			}
		}
		if !success {
			os.Exit(1)
		}
		return
	}

	if !runModelTest(*modelPath, *baselinePath, *generate, *threshold, *pplThreshold, memMgr) {
		os.Exit(1)
	}
}

func runModelTest(modelPath, baselinePath string, generate bool, threshold, pplThreshold float64, memMgr *MemoryManager) bool {
	conf := config.Default()
	fmt.Printf("Memory before NewEngine: %d bytes\n", device.AllocatedBytes())
	e, err := engine.NewEngine(modelPath, conf)
	if err != nil {
		fmt.Printf("Error creating engine: %v\n", err)
		return false
	}
	defer func() {
		e.Close()
		fmt.Printf("Memory after Close: %d bytes\n", device.AllocatedBytes())
	}()

	ggufFile, err := gguf.LoadFile(modelPath)
	if err != nil {
		fmt.Printf("Error loading GGUF for tokenizer: %v\n", err)
		return false
	}
	defer ggufFile.Close()

	tok, err := tokenizer.NewFromGGUF(ggufFile)
	if err != nil {
		fmt.Printf("Error creating tokenizer from GGUF: %v\n", err)
		return false
	}

	prompts := []string{
		"The quick brown fox",
		"Hello, my name is",
		"In a hole in the ground there lived a hobbit.",
	}

	multiTokenTests := []struct {
		prompt   string
		tokens   int
		coherent bool
	}{
		{"The capital of France is", 5, true},
		{"Once upon a time", 10, true},
		{"What is 2+2?", 3, true},
	}

	baseline := BaselineFile{
		ModelName: modelPath,
		Timestamp: time.Now().Format(time.RFC3339),
		Entries:   make(map[string]BaselineEntry),
	}

	if baselinePath == "" {
		baselinePath = fmt.Sprintf("cmd/smoke_test/baselines/%s.json", filepath.Base(modelPath))
	}

	if !generate {
		data, err := os.ReadFile(baselinePath)
		if err != nil {
			fmt.Printf("Error reading baseline: %v\n", err)
			return false
		}
		if err := json.Unmarshal(data, &baseline); err != nil {
			fmt.Printf("Error unmarshaling baseline: %v\n", err)
			return false
		}
	}

	sampler := engine.SamplerConfig{Temperature: 0}
	success := true

	for _, p := range prompts {
		fmt.Printf("Testing prompt (baseline): %q\n", p)
		inputTokens := tok.Encode(p)

		tokens, logits, err := e.InferWithLogits(inputTokens, 1, sampler)
		if err != nil {
			fmt.Printf("  Inference error: %v\n", err)
			success = false
			continue
		}

		if generate {
			sanitizedLogits := make([]float32, len(logits))
			for i, v := range logits {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					sanitizedLogits[i] = 0
				} else {
					sanitizedLogits[i] = v
				}
			}
			ppl := calculatePerplexity(logits, tokens[0])
			if math.IsNaN(ppl) || math.IsInf(ppl, 0) {
				ppl = 0
			}
			baseline.Entries[p] = BaselineEntry{
				Prompt:     p,
				Tokens:     tokens,
				Logits:     sanitizedLogits,
				Perplexity: ppl,
			}
		} else {
			entry, ok := baseline.Entries[p]
			if !ok {
				fmt.Printf("  No baseline entry for prompt\n")
				success = false
				continue
			}

			mse := calculateMSE(logits, entry.Logits)
			ppl := calculatePerplexity(logits, tokens[0])
			pplDiff := math.Abs(ppl - entry.Perplexity)

			fmt.Printf("  Logit MSE: %e\n", mse)
			fmt.Printf("  Perplexity: %.4f (diff: %.4f)\n", ppl, pplDiff)

			if mse > threshold {
				fmt.Printf("  FAILED: MSE %e exceeds threshold %e\n", mse, threshold)
				success = false
			} else if pplDiff > pplThreshold {
				fmt.Printf("  FAILED: Perplexity diff %.4f exceeds threshold %.4f\n", pplDiff, pplThreshold)
				success = false
			} else {
				fmt.Printf("  PASSED\n")
			}
		}
	}

	fmt.Println("\n=== Multi-token coherence tests ===")
	for _, test := range multiTokenTests {
		fmt.Printf("Testing prompt (coherence): %q (%d tokens)\n", test.prompt, test.tokens)
		inputTokens := tok.Encode(test.prompt)

		generatedTokens, err := e.Infer(inputTokens, test.tokens, sampler)
		if err != nil {
			fmt.Printf("  Inference error: %v\n", err)
			success = false
			continue
		}

		output := tok.Decode(generatedTokens)
		coherent := isCoherent(output)

		fmt.Printf("  Output: %q\n", output)
		if coherent {
			fmt.Printf("  Status: COHERENT\n")
		} else {
			fmt.Printf("  Status: INCOHERENT - output appears corrupted\n")
			success = false
		}
	}

	if generate {
		data, err := json.MarshalIndent(baseline, "", "  ")
		if err != nil {
			fmt.Printf("Error marshaling baseline: %v\n", err)
			return false
		}
		if baselinePath != "" {
			if err := os.WriteFile(baselinePath, data, 0644); err != nil {
				fmt.Printf("Error writing baseline: %v\n", err)
				return false
			}
			fmt.Printf("Baseline generated to %s (%d bytes)\n", baselinePath, len(data))
		} else {
			fmt.Println(string(data))
		}
	}

	return success
}
