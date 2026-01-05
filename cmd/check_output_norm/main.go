//go:build darwin && metal

package main

import (
	"fmt"
	"log"
	"math"
	"flag"

	"github.com/23skdu/longbow-quarrel/internal/engine"
)

var modelPath = flag.String("model", "", "Path to GGUF model")

func main() {
	flag.Parse()
	if *modelPath == "" {
		log.Fatal("--model required")
	}

	fmt.Println("Loading model...")
	e, err := engine.NewEngine(*modelPath, false)
	if err != nil {
		log.Fatal(err)
	}
	defer e.Close()

	fmt.Println("\n=== Checking output_norm.weight ===")
	normWeight := engine.LoadWeightFromGGUF(e, "output_norm.weight")
	
	var sum, sumSq, min, max float64
	min = math.MaxFloat64
	max = -math.MaxFloat64
	
	for _, v := range normWeight {
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
	
	mean := sum / float64(len(normWeight))
	l2norm := math.Sqrt(sumSq)
	
	fmt.Printf("Dimensions: %d\n", len(normWeight))
	fmt.Printf("L2 Norm: %.6f\n", l2norm)
	fmt.Printf("Mean: %.6f\n", mean)
	fmt.Printf("Min: %.6f\n", min)
	fmt.Printf("Max: %.6f\n", max)
	fmt.Printf("Sample (first 10): %v\n", normWeight[:10])
	
	// Check if all values are close to 1.0 (expected for RMSNorm weights)
	var deviations int
	for _, v := range normWeight {
		if math.Abs(float64(v) - 1.0) > 0.5 {
			deviations++
		}
	}
	
	fmt.Printf("\nValues deviating >0.5 from 1.0: %d / %d (%.2f%%)\n", 
		deviations, len(normWeight), float64(deviations)/float64(len(normWeight))*100)
	
	if deviations > len(normWeight)/10 {
		fmt.Println("⚠️  WARNING: Many values deviate from 1.0 - this may indicate an issue!")
	} else {
		fmt.Println("✓ Most values are close to 1.0")
	}
}
