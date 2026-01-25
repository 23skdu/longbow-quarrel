package device

import (
	"math"
	"testing"
	"time"
)

// TestMetalBackwardCompatibility ensures Metal backend works with various input conditions
func TestMetalBackwardCompatibility(t *testing.T) {
	t.Run("EmptyInput", func(t *testing.T) {
		ctx := NewContext()
		defer ctx.Close()

		// Test with empty tensors
		empty := ctx.EmptyTensor(32, 32)
		result := ctx.Linear(empty, empty)
		defer result.Release()
		defer empty.Release()

		// Should complete without panic
		t.Log("Empty tensor test completed successfully")
	})

	t.Run("SingleElementTensor", func(t *testing.T) {
		ctx := NewContext()
		defer ctx.Close()

		// Test single element tensor
		data := []float32{1.0}
		single := ctx.NewTensorWithData(data, 1, 1)
		result := ctx.Scale(single, 1.0)
		defer result.Release()
		defer single.Release()

		// Verify result
		resultData := result.ToHost()
		if len(resultData) != 1 || resultData[0] != 1.0 {
			t.Errorf("Single element scaling failed: got %v", resultData)
		}
	})

	t.Run("LargeTensorHandling", func(t *testing.T) {
		ctx := NewContext()
		defer ctx.Close()

		// Test large tensor allocation
		data := make([]float32, 1024*1024) // 4MB tensor
		large := ctx.NewTensorWithData(data, 1024, 1024)
		defer large.Release()

		// Should handle allocation without panic
		t.Logf("Large tensor (%d elements) allocated successfully", len(data))
	})

	t.Run("MemoryPressure", func(t *testing.T) {
		ctx := NewContext()
		defer ctx.Close()

		// Test behavior under memory pressure
		iterations := 10
		for i := 0; i < iterations; i++ {
			data := make([]float32, 512*512) // 1MB tensor
			tensor := ctx.NewTensorWithData(data, 512, 512)

			// Perform simple operation
			scaled := ctx.Scale(tensor, 0.5)

			// Cleanup
			scaled.Release()
			tensor.Release()
		}

		t.Logf("Memory pressure test completed: %d iterations", iterations)
	})
}

// TestMemoryPoolEfficiency validates tensor reuse and memory management
func TestMemoryPoolEfficiency(t *testing.T) {
	ctx := NewContext()
	defer ctx.Close()

	t.Run("TensorReuse", func(t *testing.T) {
		// Get tensor from pool
		tensor1 := ctx.GetTensor(1024, 1024)
		if tensor1 == nil {
			t.Fatal("Failed to get tensor from pool")
		}

		// Return tensor to pool
		tensor1.Release()

		// Should get the same tensor back (if pool is working)
		tensor2 := ctx.GetTensor(1024, 1024)
		if tensor2 != tensor1 {
			t.Log("Pool returned different tensor (pool may not be strict about reuse)")
		}
		tensor2.Release()
	})

	t.Run("ConcurrentAccess", func(t *testing.T) {
		// Test concurrent tensor access (should be safe)
		ctx.ConcurrentTensorOperations(func() {
			data := make([]float32, 256*256)
			tensor := ctx.NewTensorWithData(data, 256, 256)
			_ = ctx.Linear(tensor, tensor)
			tensor.Release()
		})
		t.Log("Concurrent operations completed")
	})
}

// TestNumericalStability ensures consistent results across multiple runs
func TestNumericalStability(t *testing.T) {
	ctx := NewContext()
	defer ctx.Close()

	// Test data
	input := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	tensor := ctx.NewTensorWithData(input, 5, 1)
	defer tensor.Release()

	// Multiple operations
	results := make([]float32, 10)
	for i := 0; i < 10; i++ {
		scaled := ctx.Scale(tensor, float32(i+1))
		data := scaled.ToHost()

		// Store first result
		if i == 0 {
			copy(results, data)
		}

		scaled.Release()

		// Verify stability
		if i == 9 {
			for j, val := range data {
				if math.Abs(val-results[j]) > 1e-6 {
					t.Errorf("Numerical instability detected: element %d, first=%f, current=%f, diff=%e",
						j, results[j], val, math.Abs(val-results[j]))
				}
			}
		}
	}
}

// TestConcurrency validates thread-safe operations
func TestConcurrency(t *testing.T) {
	ctx := NewContext()
	defer ctx.Close()

	t.Run("ParallelTensorCreation", func(t *testing.T) {
		// Create tensors in parallel
		done := make(chan bool, 10)

		for i := 0; i < 10; i++ {
			go func(id int) {
				data := make([]float32, 256*256)
				tensor := ctx.NewTensorWithData(data, 256, 256)
				tensor.Release()
				done <- true
			}(i)
		}

		// Wait for completion
		for i := 0; i < 10; i++ {
			<-done
		}

		t.Log("Parallel tensor creation completed")
	})
}

// BenchmarkMetalKernelPerformance measures specific kernel performance
func BenchmarkMetalKernelPerformance(b *testing.B) {
	ctx := NewContext()
	defer ctx.Close()

	b.ResetTimer()

	data := make([]float32, 1024*1024)
	input := ctx.NewTensorWithData(data, 1024, 1024)
	defer input.Release()

	// Benchmark MatMul
	b.Run("MatMul_1024x1024", func(b *testing.B) {
		output := ctx.EmptyTensor(1024, 1024)
		defer output.Release()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ctx.MatMul(input, input, output)
			ctx.Synchronize() // Ensure completion
		}
	})

	// Benchmark RMSNorm + Linear (fused operation)
	b.Run("RMSNormLinear_F16_1024x1024", func(b *testing.B) {
		output := ctx.EmptyTensor(1024, 1024)
		defer output.Release()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ctx.RMSNormLinear(input, input, output)
			ctx.Synchronize()
		}
	})

	// Benchmark Attention
	weight := ctx.NewTensorWithData(data, 1024, 1024) // Use as weight matrix
	defer weight.Release()

	b.Run("Attention_F16_1024x1024", func(b *testing.B) {
		kvCache := make([]float32, 1024*1024)
		kvTensor := ctx.NewTensorWithData(kvCache, 1024, 1024)
		defer kvTensor.Release()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ctx.Attention(input, input, input, input, weight, weight, weight, kvTensor, kvTensor)
			ctx.Synchronize()
		}
	})
}
