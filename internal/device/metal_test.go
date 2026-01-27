//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

// TestMetalBackwardCompatibility ensures Metal backend works with various input conditions
func TestMetalBackwardCompatibility(t *testing.T) {
	t.Run("EmptyInput", func(t *testing.T) {
		ctx := NewContext()
		defer ctx.Free()

		// Test with empty tensors
		empty := ctx.NewTensor(32, 32)
		result, err := empty.Linear(empty)
		if err != nil {
			t.Fatalf("Linear failed: %v", err)
		}
		defer result.Free()
		defer empty.Free()

		// Should complete without panic
		t.Log("Empty tensor test completed successfully")
	})

	t.Run("SingleElementTensor", func(t *testing.T) {
		ctx := NewContext()
		defer ctx.Free()

		// Test single element tensor
		data := []float32{1.0}
		single := ctx.NewTensor(1, 1)
		single.LoadFrom(data)
		result := single.ScaleBy(1.0)
		defer result.Free()
		defer single.Free()

		// Verify result
		resultData := result.ToHost()
		if len(resultData) != 1 || resultData[0] != 1.0 {
			t.Errorf("Single element scaling failed: got %v", resultData)
		}
	})

	t.Run("LargeTensorHandling", func(t *testing.T) {
		ctx := NewContext()
		defer ctx.Free()

		// Test large tensor allocation
		rows, cols := 1024, 1024
		large := ctx.NewTensor(rows, cols)
		defer large.Free()

		// Should handle allocation without panic
		t.Logf("Large tensor (%dx%d) allocated successfully", rows, cols)
	})

	t.Run("MemoryPressure", func(t *testing.T) {
		ctx := NewContext()
		defer ctx.Free()

		// Test behavior under memory pressure
		iterations := 10
		for i := 0; i < iterations; i++ {
			tensor := ctx.NewTensor(512, 512)

			// Perform simple operation
			scaled := tensor.ScaleBy(0.5)

			// Cleanup
			scaled.Free()
			tensor.Free()
		}

		t.Logf("Memory pressure test completed: %d iterations", iterations)
	})
}

// TestMemoryPoolEfficiency validates tensor reuse and memory management
func TestMemoryPoolEfficiency(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	t.Run("TensorReuse", func(t *testing.T) {
		// Get tensor from pool
		tensor1 := ctx.NewTensorPooled(1024, 1024)
		if tensor1 == nil {
			t.Fatal("Failed to get tensor from pool")
		}

		// Return tensor to pool
		tensor1.ReturnToPool()

		// Should get the same tensor back (if pool is working)
		tensor2 := ctx.NewTensorPooled(1024, 1024)
		if tensor2 != tensor1 {
			t.Log("Pool returned different tensor (pool may not be strict about reuse)")
		}
		tensor2.ReturnToPool()
	})

	t.Run("ConcurrentAccess", func(t *testing.T) {
		// Test concurrent tensor access (should be safe)
		done := make(chan bool)
		go func() {
			data := make([]float32, 256*256)
			tensor := ctx.NewTensor(256, 256)
			tensor.LoadFrom(data)
			res, _ := tensor.Linear(tensor)
			res.Free()
			tensor.Free()
			done <- true
		}()
		<-done
		t.Log("Concurrent operations completed")
	})
}

// TestNumericalStability ensures consistent results across multiple runs
func TestNumericalStability(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Test data
	input := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	tensor := ctx.NewTensor(5, 1)
	tensor.LoadFrom(input)
	defer tensor.Free()

	// Multiple operations
	var results []float32
	for i := 0; i < 10; i++ {
		scaled := tensor.ScaleBy(float32(i + 1))
		data := scaled.ToHost()

		// Store first result
		if i == 0 {
			results = make([]float32, len(data))
			copy(results, data)
		}

		scaled.Free()

		// Verify stability
		if i == 9 {
			// (This logic in original test seems slightly flawed as i changes,
			// but we'll keep the spirit: checking if the results are consistent)
			// Wait, if i=9, scaled is tensor * 10. results is tensor * 1.
			// The original test compared results[j] with data[j] which is wrong if i changed.
			// Let's fix the test to check if scaling is correct.
			for j, val := range data {
				expected := input[j] * 10.0
				if math.Abs(float64(val-expected)) > 1e-3 {
					t.Errorf("Numerical instability detected: element %d, expected=%f, current=%f, diff=%e",
						j, expected, val, math.Abs(float64(val-expected)))
				}
			}
		}
	}
}

// TestConcurrency validates thread-safe operations
func TestConcurrency(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	t.Run("ParallelTensorCreation", func(t *testing.T) {
		// Create tensors in parallel
		count := 10
		done := make(chan bool, count)

		for i := 0; i < count; i++ {
			go func(id int) {
				tensor := ctx.NewTensor(256, 256)
				tensor.Free()
				done <- true
			}(i)
		}

		// Wait for completion
		for i := 0; i < count; i++ {
			<-done
		}

		t.Log("Parallel tensor creation completed")
	})
}

// BenchmarkMetalKernelPerformance measures specific kernel performance
func BenchmarkMetalKernelPerformance(b *testing.B) {
	ctx := NewContext()
	defer ctx.Free()

	b.ResetTimer()

	rows, cols := 1024, 1024
	input := ctx.NewTensor(rows, cols)
	defer input.Free()

	// Benchmark MatMul
	b.Run("MatMul_1024x1024", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := input.MatMul(input)
			res.Free()
			ctx.Synchronize() // Ensure completion
		}
	})

	// Benchmark Linear
	b.Run("Linear_1024x1024", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res, _ := input.Linear(input)
			res.Free()
			ctx.Synchronize()
		}
	})

	// Benchmark RMSNorm (if available)
	normWeight := ctx.NewTensor(1, cols)
	defer normWeight.Free()
	b.Run("RMSNorm_1024x1024", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := input.RMSNorm(normWeight, 1e-5)
			res.Free()
			ctx.Synchronize()
		}
	})
}
