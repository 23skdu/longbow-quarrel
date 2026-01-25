package gguf

import "fmt"

func QuantizeWeightsToQ4K(weights []float32, numElements int) ([]byte, error) {
	return nil, fmt.Errorf("not implemented")
}

func DequantizeWeightsFromQ4K(data []byte, rows, cols int) ([]float32, error) {
	numElements := rows * cols
	return DequantizeQ4K(data, numElements), nil
}
