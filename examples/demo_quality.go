package main

import (
	"fmt"

	"github.com/23skdu/longbow-quarrel/internal/engine"
)

func main() {
	// Create quality evaluator
	eval := engine.NewQualityEvaluatorSimple()

	// Test texts
	reference := "The weather today is beautiful and sunny"
	candidates := []string{
		"The weather today is beautiful and sunny", // Perfect match
		"The weather is nice and sunny today",      // Good match
		"The cat sat on the mat",                   // Different topic
		"Weather today beautiful sunny",            // Missing words
	}

	fmt.Println("=== Quality Metrics Demonstration ===\n")
	fmt.Printf("Reference: %q\n\n", reference)

	for i, candidate := range candidates {
		fmt.Printf("Candidate %d: %q\n", i+1, candidate)

		// Calculate BLEU score
		bleu := eval.CalculateBLEU(candidate, reference)
		fmt.Printf("  BLEU-4: %.4f (Precision: %.2f, %.2f, %.2f, %.2f)\n",
			bleu.BLEU4, bleu.Precision[0], bleu.Precision[1], bleu.Precision[2], bleu.Precision[3])

		// Calculate ROUGE score
		rouge := eval.CalculateROUGE(candidate, reference)
		fmt.Printf("  ROUGE-1 F1: %.4f (Precision: %.4f, Recall: %.4f)\n",
			rouge.F1, rouge.Precision, rouge.Recall)

		// Calculate perplexity (simplified)
		// Note: This would need actual token IDs in a real implementation
		tokens := []int{1, 2, 3, 4, 5} // Placeholder tokens
		pplx := eval.CalculatePerplexity(tokens)
		fmt.Printf("  Perplexity: %.2f (simplified demo)\n", pplx.Perplexity)

		fmt.Println()
	}

	fmt.Println("=== Notes ===")
	fmt.Println("- BLEU measures n-gram overlap (higher = more similar)")
	fmt.Println("- ROUGE measures recall and precision of overlapping units")
	fmt.Println("- Perplexity measures how 'surprised' the model is (lower = better)")
	fmt.Println("- This demo uses simplified character-based calculations")
	fmt.Println("- Production use would require proper tokenization and model-based perplexity")
}
