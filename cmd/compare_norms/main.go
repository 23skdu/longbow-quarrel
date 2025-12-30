package main

import (
	"fmt"
	"log"
	"math"
	"flag"

	"github.com/23skdu/longbow-quarrel/internal/engine"
)

var modelPath = flag.String("model", "", "Path to GGUF model")

func analyzeNorm(e *engine.Engine, name string) {
	weights := engine.LoadWeightFromGGUF(e, name)
	
	var sum, sumSq, min, max float64
	min = math.MaxFloat64
	max = -math.MaxFloat64
	
	for _, v := range weights {
		val := float64(v)
		sum += val
		sumSq += val * val
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}
	
	mean := sum / float64(len(weights))
	l2norm := math.Sqrt(sumSq)
	
	fmt.Printf("%-30s: Mean=%.4f L2=%.4f Min=%.4f Max=%.4f\n", 
		name, mean, l2norm, min, max)
}

func main() {
	flag.Parse()
	if *modelPath == "" {
		log.Fatal("--model required")
	}

	e, err := engine.NewEngine(*modelPath, false)
	if err != nil {
		log.Fatal(err)
	}
	defer e.Close()

	fmt.Println("=== Comparing All Norm Weights ===\n")
	
	// Check output norm
	analyzeNorm(e, "output_norm.weight")
	
	// Check a few layer norms
	for i := 0; i < 5; i++ {
		analyzeNorm(e, fmt.Sprintf("blk.%d.attn_norm.weight", i))
	}
	
	for i := 0; i < 5; i++ {
		analyzeNorm(e, fmt.Sprintf("blk.%d.ffn_norm.weight", i))
	}
}
