package main

import (
	"fmt"
)

func main() {
	fmt.Println("=== Longbow Quarrel Streaming API Demo ===")

	// Note: This is a simplified demo. In practice, you would:
	// 1. Load a real model via Ollama or GGUF file
	// 2. Use a proper tokenizer
	// 3. Handle model initialization

	fmt.Println("Streaming API Features:")
	fmt.Println("✅ Real-time token generation with callbacks")
	fmt.Println("✅ Immediate output as tokens are produced")
	fmt.Println("✅ Compatible with all sampling modes")
	fmt.Println("✅ No performance penalty for non-streaming use")
	fmt.Println()

	fmt.Println("Usage Examples:")
	fmt.Println()
	fmt.Println("# Basic streaming")
	fmt.Println("./quarrel -model smollm2:135m-instruct-fp16 -prompt \"Hello\" -n 10 -stream")
	fmt.Println()
	fmt.Println("# Streaming with quality sampling")
	fmt.Println("./quarrel -model smollm2:135m-instruct-fp16 -prompt \"Hello\" -n 10 -stream -quality")
	fmt.Println()
	fmt.Println("# Streaming with custom sampling parameters")
	fmt.Println("./quarrel -model smollm2:135m-instruct-fp16 -prompt \"Hello\" -n 10 -stream -temp 0.8 -topp 0.9")
	fmt.Println()

	fmt.Println("API Usage in Code:")
	fmt.Println(`
// Streaming inference with callback
result, err := engine.InferWithCallback(inputTokens, numTokens, config, func(token int) {
    decoded := tokenizer.Decode([]int{token})
    fmt.Print(decoded) // Output immediately
})

// Regular inference (no streaming)
result, err := engine.Infer(inputTokens, numTokens, config)
`)
	fmt.Println()
	fmt.Println("Benefits:")
	fmt.Println("🎯 Better user experience with immediate feedback")
	fmt.Println("⚡ Faster apparent response times")
	fmt.Println("🔄 Compatible with existing chat applications")
	fmt.Println("📊 No impact on generation quality or speed")
}
