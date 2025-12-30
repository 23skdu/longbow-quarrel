package main

import (
	"fmt"
	"log"
	"math"
	"flag"
	"sort"

	"github.com/23skdu/longbow-quarrel/internal/engine"
)

var modelPath = flag.String("model", "", "Path to GGUF model")

type TokenStats struct {
	TokenID int
	L2Norm  float64
	Mean    float64
	Max     float64
	Min     float64
}

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

	fmt.Println("Loading output.weight...")
	outputWeight := engine.LoadWeightFromGGUF(e, "output.weight")
	
	// output.weight is [VocabSize, HiddenDim]
	vocabSize := e.Config.VocabSize
	hiddenDim := e.Config.Dim
	
	fmt.Printf("Analyzing %d tokens x %d dimensions\n", vocabSize, hiddenDim)
	
	// Compute statistics for each token
	stats := make([]TokenStats, vocabSize)
	
	for tokenID := 0; tokenID < vocabSize; tokenID++ {
		rowStart := tokenID * hiddenDim
		rowEnd := rowStart + hiddenDim
		row := outputWeight[rowStart:rowEnd]
		
		var sum, sumSq, min, max float64
		min = math.MaxFloat64
		max = -math.MaxFloat64
		
		for _, v := range row {
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
		
		mean := sum / float64(hiddenDim)
		l2norm := math.Sqrt(sumSq)
		
		stats[tokenID] = TokenStats{
			TokenID: tokenID,
			L2Norm:  l2norm,
			Mean:    mean,
			Max:     max,
			Min:     min,
		}
	}
	
	// Find tokens with highest/lowest L2 norms
	sortedByNorm := make([]TokenStats, len(stats))
	copy(sortedByNorm, stats)
	sort.Slice(sortedByNorm, func(i, j int) bool {
		return sortedByNorm[i].L2Norm > sortedByNorm[j].L2Norm
	})
	
	fmt.Println("\n=== Top 10 Tokens by L2 Norm ===")
	for i := 0; i < 10 && i < len(sortedByNorm); i++ {
		s := sortedByNorm[i]
		fmt.Printf("Token %5d: L2=%.4f Mean=%.6f Min=%.4f Max=%.4f\n",
			s.TokenID, s.L2Norm, s.Mean, s.Min, s.Max)
	}
	
	fmt.Println("\n=== Bottom 10 Tokens by L2 Norm ===")
	for i := len(sortedByNorm) - 10; i < len(sortedByNorm); i++ {
		if i < 0 {
			continue
		}
		s := sortedByNorm[i]
		fmt.Printf("Token %5d: L2=%.4f Mean=%.6f Min=%.4f Max=%.4f\n",
			s.TokenID, s.L2Norm, s.Mean, s.Min, s.Max)
	}
	
	// Check token 16277 specifically
	fmt.Println("\n=== Token 16277 (omitempty) ===")
	if 16277 < len(stats) {
		s := stats[16277]
		fmt.Printf("Token %5d: L2=%.4f Mean=%.6f Min=%.4f Max=%.4f\n",
			s.TokenID, s.L2Norm, s.Mean, s.Min, s.Max)
		
		// Find its rank
		rank := 0
		for i, st := range sortedByNorm {
			if st.TokenID == 16277 {
				rank = i + 1
				break
			}
		}
		fmt.Printf("Rank: %d / %d (%.2f percentile)\n", 
			rank, vocabSize, float64(rank)/float64(vocabSize)*100)
	}
	
	// Compute overall statistics
	var totalNorm, totalMean float64
	for _, s := range stats {
		totalNorm += s.L2Norm
		totalMean += s.Mean
	}
	avgNorm := totalNorm / float64(vocabSize)
	avgMean := totalMean / float64(vocabSize)
	
	fmt.Printf("\n=== Overall Statistics ===\n")
	fmt.Printf("Average L2 Norm: %.4f\n", avgNorm)
	fmt.Printf("Average Mean: %.6f\n", avgMean)
	
	// Check if token 16277 is an outlier
	if 16277 < len(stats) {
		s := stats[16277]
		normRatio := s.L2Norm / avgNorm
		fmt.Printf("\nToken 16277 L2 Norm Ratio: %.2fx average\n", normRatio)
		if normRatio > 2.0 {
			fmt.Println("⚠️  WARNING: Token 16277 has abnormally HIGH L2 norm!")
		} else if normRatio < 0.5 {
			fmt.Println("⚠️  WARNING: Token 16277 has abnormally LOW L2 norm!")
		} else {
			fmt.Println("✓ Token 16277 L2 norm is within normal range")
		}
	}
}
