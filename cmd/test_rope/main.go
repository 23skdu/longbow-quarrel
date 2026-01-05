//go:build darwin && metal

package main

import (
	"fmt"
	"log"
	"math"
	"flag"

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/engine"
)

var modelPath = flag.String("model", "", "Path to GGUF model")

func main() {
	flag.Parse()
	if *modelPath == "" {
		log.Fatal("--model required")
	}

	// Load Engine
	fmt.Println("Loading Engine...")
	e, err := engine.NewEngine(*modelPath, false)
	if err != nil {
		log.Fatal(err)
	}
	defer e.Close()

	// Test RoPE at different positions
	testRoPEAtPosition(e, 0)
	testRoPEAtPosition(e, 1)
	testRoPEAtPosition(e, 5)
}

func testRoPEAtPosition(e *engine.Engine, pos int) {
	fmt.Printf("\n=== Testing RoPE at Pos=%d ===\n", pos)
	
	// Create test input
	dim := e.Config.Dim
	input := make([]float32, dim)
	for i := range input {
		input[i] = float32(math.Sin(float64(i)*0.1)) * 0.1
	}
	
	// GPU Test
	tInput := e.Ctx.NewTensor(1, dim)
	tInput.LoadFrom(input)
	
	// Apply RoPE
	tInput.RoPE(pos, e.Config.HeadDim, e.Config.Heads, 1, e.Config.RopeTheta)
	e.Ctx.Synchronize()
	
	gpuResult := tInput.ToHost()
	tInput.Free()
	
	// CPU Reference
	cpuResult := device.CPURoPE(input, pos, e.Config.Heads, e.Config.HeadDim, e.Config.RopeTheta)
	
	// Compare
	var maxDiff float64
	var sumDiff float64
	for i := range input {
		diff := math.Abs(float64(gpuResult[i] - cpuResult[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		sumDiff += diff
	}
	avgDiff := sumDiff / float64(len(input))
	
	fmt.Printf("MaxDiff: %.6f, AvgDiff: %.6f\n", maxDiff, avgDiff)
	
	if maxDiff > 1e-2 {
		fmt.Printf("FAIL: RoPE mismatch at Pos=%d\n", pos)
		fmt.Printf("Sample (first 10):\n")
		for i := 0; i < 10; i++ {
			fmt.Printf("  [%d] CPU: %.6f, GPU: %.6f, Diff: %.6f\n", 
				i, cpuResult[i], gpuResult[i], math.Abs(float64(gpuResult[i]-cpuResult[i])))
		}
	} else {
		fmt.Printf("PASS: RoPE matches at Pos=%d\n", pos)
	}
}
