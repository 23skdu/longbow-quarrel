//go:build darwin && metal

package device

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func TestLinearQ4K_AttentionProjections(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Mistral Dimensions
	// Hidden Size: 4096
	// KV Proj Size: 1024 (8 * 128)
	
	cols := 4096 // Input dimension
	rows := 1024 // Output dimension
	
	// 1. Generate Q4_K Weights
	numBlocks := (rows * cols) / 256
	dataSize := numBlocks * 144
	q4kData := make([]byte, dataSize)
	rand.Seed(time.Now().UnixNano())
	rand.Read(q4kData) 
	
	for i := 0; i < numBlocks; i++ {
		offset := i * 144
		// Valid scales/mins
		d := Float32ToFloat16(rand.Float32() * 0.05)
		dmin := Float32ToFloat16(rand.Float32() * 0.01)
		binary.LittleEndian.PutUint16(q4kData[offset:], d)
		binary.LittleEndian.PutUint16(q4kData[offset+2:], dmin)
	}

	// 2. CPU Reference
	t.Log("Dequantizing Q4K weights for CPU reference...")
	weightsF32 := gguf.DequantizeQ4K(q4kData, rows*cols)
	
	// Input Vector (1 token)
	inputF32 := make([]float32, cols)
	for i := range inputF32 {
		inputF32[i] = (rand.Float32() - 0.5) * 40.0 // Larger range [-20, 20] for Mistral stability check
	}
	
	// CPU MatMul
	expected := make([]float32, rows)
	for r := 0; r < rows; r++ {
		sum := float32(0.0)
		rowOffset := r * cols
		for c := 0; c < cols; c++ {
			sum += weightsF32[rowOffset+c] * inputF32[c]
		}
		expected[r] = sum
	}

	// 3. GPU Execution
	wTen := ctx.NewQ4KTensor(rows, cols)
	wTen.LoadFromRaw(q4kData)
	
	inTen := ctx.NewTensor(1, cols)
	inTen.LoadFrom(inputF32)
	
	// Check Q4K MatMul
	outTen := wTen.MatMul(inTen)
	if err := ctx.WaitWithTimeout(5 * time.Second); err != nil {
		t.Fatal(err)
	}
	
	res := outTen.ToHost()
	
	wTen.Free()
	inTen.Free()
	outTen.Free()
	
	// 4. Validate output
	if len(res) != rows {
		t.Fatalf("Expected %d result elements, got %d", rows, len(res))
	}
	
	maxDiff := float32(0.0)
	avgDiff := float32(0.0)
	maxVal := float32(0.0)
	
	for i := 0; i < rows; i++ {
		val := float32(math.Abs(float64(expected[i])))
		if val > maxVal {
			maxVal = val
		}
		
		diff := float32(math.Abs(float64(res[i] - expected[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
		avgDiff += diff
	}
	avgDiff /= float32(rows)
	
	t.Logf("Max Val: %f", maxVal)
	t.Logf("Max Diff: %f", maxDiff)
	t.Logf("Avg Diff: %f", avgDiff)
	
	// Check relative error
    // For large values, absolute diff can be large due to FP associativity.
    // 0.5% relative error map be acceptable for Q4_K matrix multiplication
    // on GPU (fast math) vs CPU (precise).
    
    var relativeError float32
    if maxVal > 1e-6 {
        relativeError = maxDiff / maxVal
    }
    
    t.Logf("Relative Error: %f%%", relativeError * 100)

	if relativeError > 0.01 { // 1% tolerance
		t.Errorf("Precision failure! Relative error %f%% exceeds 1%%", relativeError * 100)
	}
}
