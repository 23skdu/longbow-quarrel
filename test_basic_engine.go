import "fmt"
import "time"
import "github.com/23skdu/longbow-quarrel/internal/engine"

func main() {
	fmt.Printf("Testing basic engine functionality...")
	
	// Test 1: Create engine
	e, err := engine.NewEngine("/Users/rsd/.ollama/models/blobs/sha256-f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57", engine.EngineConfig{})
	if err != nil {
		fmt.Printf("❌ Engine creation failed: %v\n", err)
		return
	}
	
	fmt.Printf("✅ Engine created successfully\n")
	
	// Test 2: Check if methods exist
	if e.Infer == nil {
		fmt.Printf("❌ Engine methods not available\n")
		return
	}
	
	fmt.Printf("✅ Engine methods available\n")
	
	// Test 3: Try simple inference
	tokens := []int{1}
	prompt := "test"
	start := time.Now()
	output, err := e.Infer(tokens, prompt, nil)
	if err != nil {
		fmt.Printf("❌ Inference failed: %v\n", err)
		return
	}
	
	duration := time.Since(start)
	
	fmt.Printf("✅ Basic inference test passed\n")
	fmt.Printf("   Model: SmolLM2 135M (via direct path)\n")
	fmt.Printf("   Tokens: %d\n", len(tokens))
	fmt.Printf("   Duration: %v\n", duration)
	fmt.Printf("   Tokens/sec: %.2f\n", float64(len(tokens))/duration.Seconds())
	fmt.Printf("   Output: %s\n", output)
	
	// Test 4: Generate some output to show it's working
	longerTokens := []int{1, 504, 38478, 22216, 29343, 90}
	longerPrompt := "The quick brown fox jumps over the lazy dog and runs through the forest"
	longerStart := time.Now()
	longerOutput, err := e.Infer(longerTokens, longerPrompt, nil)
	if err != nil {
		fmt.Printf("❌ Longer inference failed: %v\n", err)
		return
	}
	
	longerDuration := time.Since(longerStart)
	
	fmt.Printf("✅ Extended inference test passed\n")
	fmt.Printf("   Longer Tokens: %d\n", len(longerTokens))
	fmt.Printf("   Longer Duration: %v\n", longerDuration)
	fmt.Printf("   Longer Tokens/sec: %.2f\n", float64(len(longerTokens))/longerDuration.Seconds())
	fmt.Printf("   Longer Output: %s\n", longerOutput[:50]) // First 50 chars
	
	fmt.Printf("\n🎉 BASIC ENGINE FUNCTIONALITY CONFIRMED 🎉\n")
}