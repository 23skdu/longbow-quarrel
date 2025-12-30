package main

import (
	"encoding/json"
	"fmt"
	"os"
	"github.com/23skdu/longbow-quarrel/internal/engine"
)

// ActivationDump stores layer-by-layer activations for comparison
type ActivationDump struct {
	Prompt      string                   `json:"prompt"`
	Tokens      []int                    `json:"tokens"`
	Embedding   []float32                `json:"embedding"`
	Layers      []LayerActivations       `json:"layers"`
	FinalLogits []float32                `json:"final_logits"`
}

type LayerActivations struct {
	LayerIdx        int       `json:"layer_idx"`
	InputMax        float32   `json:"input_max"`
	QMax            float32   `json:"q_max"`
	KMax            float32   `json:"k_max"`
	VMax            float32   `json:"v_max"`
	AttentionMax    float32   `json:"attention_max"`
	FFNGateMax      float32   `json:"ffn_gate_max"`
	FFNUpMax        float32   `json:"ffn_up_max"`
	OutputMax       float32   `json:"output_max"`
	// Sample values for detailed comparison
	QSample         []float32 `json:"q_sample"`
	KSample         []float32 `json:"k_sample"`
	VSample         []float32 `json:"v_sample"`
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run cmd/activation_dump/main.go <prompt>")
		fmt.Println("Example: go run cmd/activation_dump/main.go \"The capital of France is\"")
		os.Exit(1)
	}
	
	prompt := os.Args[1]
	modelPath := "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
	
	fmt.Printf("Loading model: %s\n", modelPath)
	e, err := engine.NewEngine(modelPath, false)
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		os.Exit(1)
	}
	
	fmt.Printf("Tokenizing prompt: %q\n", prompt)
	tokens := e.Tokenizer.Encode(prompt)
	fmt.Printf("Tokens: %v\n", tokens)
	
	// TODO: Add hooks to engine to capture activations
	// For now, this is a placeholder showing the structure
	
	dump := ActivationDump{
		Prompt: prompt,
		Tokens: tokens,
		Layers: make([]LayerActivations, 32), // Mistral has 32 layers
	}
	
	// Output to JSON
	outputFile := "activations_dump.json"
	data, err := json.MarshalIndent(dump, "", "  ")
	if err != nil {
		fmt.Printf("Error marshaling JSON: %v\n", err)
		os.Exit(1)
	}
	
	err = os.WriteFile(outputFile, data, 0644)
	if err != nil {
		fmt.Printf("Error writing file: %v\n", err)
		os.Exit(1)
	}
	
	fmt.Printf("Activation dump written to: %s\n", outputFile)
	fmt.Println("\nTo compare with llama.cpp:")
	fmt.Println("1. Run llama.cpp with --log-disable and custom logging")
	fmt.Println("2. Extract activation values at each layer")
	fmt.Println("3. Compare JSON outputs to find divergence point")
}
