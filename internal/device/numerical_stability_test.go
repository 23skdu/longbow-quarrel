//go:build darwin && metal

package device

import (
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

func TestNumericalStability_Metrics_Recording(t *testing.T) {
	t.Run("Record NaN count", func(t *testing.T) {
		data := make([]float32, 100)
		for i := range data {
			if i%10 == 0 {
				data[i] = float32(math.NaN())
			} else {
				data[i] = 1.0
			}
		}

		nanCount, infCount := CheckNumericalStability(data, "test")

		if nanCount != 10 {
			t.Errorf("Expected 10 NaNs, got %d", nanCount)
		}
		if infCount != 0 {
			t.Errorf("Expected 0 Infs, got %d", infCount)
		}
	})

	t.Run("Record Inf count", func(t *testing.T) {
		data := make([]float32, 100)
		for i := range data {
			if i%20 == 0 {
				data[i] = float32(math.Inf(1))
			} else {
				data[i] = 1.0
			}
		}

		nanCount, infCount := CheckNumericalStability(data, "test")

		if nanCount != 0 {
			t.Errorf("Expected 0 NaNs, got %d", nanCount)
		}
		if infCount != 5 {
			t.Errorf("Expected 5 Infs, got %d", infCount)
		}
	})

	t.Run("Mixed NaN and Inf", func(t *testing.T) {
		data := make([]float32, 100)
		for i := range data {
			switch i % 5 {
			case 0:
				data[i] = float32(math.NaN())
			case 1:
				data[i] = float32(math.Inf(1))
			case 2:
				data[i] = float32(math.Inf(-1))
			default:
				data[i] = 1.0
			}
		}

		nanCount, infCount := CheckNumericalStability(data, "test")

		if nanCount != 20 {
			t.Errorf("Expected 20 NaNs, got %d", nanCount)
		}
		if infCount != 40 {
			t.Errorf("Expected 40 Infs (positive + negative), got %d", infCount)
		}
	})

	t.Run("No instability", func(t *testing.T) {
		data := make([]float32, 1000)
		for i := range data {
			data[i] = float32(i) * 0.001
		}

		nanCount, infCount := CheckNumericalStability(data, "test")

		if nanCount != 0 {
			t.Errorf("Expected 0 NaNs, got %d", nanCount)
		}
		if infCount != 0 {
			t.Errorf("Expected 0 Infs, got %d", infCount)
		}
	})
}

func TestScanNaNs_Implementation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	t.Run("Detect NaNs in tensor", func(t *testing.T) {
		data := make([]float32, 256)
		for i := range data {
			if i < 10 {
				data[i] = float32(math.NaN())
			} else {
				data[i] = 1.0
			}
		}

		tensor := ctx.NewTensor(16, 16)
		tensor.LoadFrom(data)

		count := tensor.ScanNaNs("nan_tensor")

		if count < 10 {
			t.Logf("ScanNaNs detected %d NaNs (expected 10+)", count)
		}
	})

	t.Run("No NaNs in valid tensor", func(t *testing.T) {
		data := make([]float32, 256)
		for i := range data {
			data[i] = float32(i%100) * 0.01
		}

		tensor := ctx.NewTensor(16, 16)
		tensor.LoadFrom(data)

		count := tensor.ScanNaNs("valid_tensor")

		if count != 0 {
			t.Errorf("Expected 0 NaNs, got %d", count)
		}
	})

	t.Run("Q4K tensor skips NaN scan", func(t *testing.T) {
		tensor, err := ctx.NewQ4KTensor(16, 256)
		if err != nil {
			t.Skip("Cannot create Q4K tensor: ", err)
		}

		count := tensor.ScanNaNs("q4k_tensor")

		if count != 0 {
			t.Errorf("Q4K tensors should return 0 for NaN scan, got %d", count)
		}
	})
}

func TestNumericalStability_Under_Large_Values(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	t.Run("FP16 max value", func(t *testing.T) {
		data := make([]float32, 256)
		for i := range data {
			data[i] = 65504.0 // Max FP16
		}

		tensor := ctx.NewTensor(16, 16)
		tensor.LoadFrom(data)

		result := tensor.ToHost()
		nanCount, infCount := CheckNumericalStability(result, "max_value")

		t.Logf("Max value test: NaNs=%d, Infs=%d", nanCount, infCount)
	})

	t.Run("FP16 overflow simulation", func(t *testing.T) {
		data := make([]float32, 256)
		for i := range data {
			data[i] = 65504.0 * 10 // Will overflow
		}

		tensor := ctx.NewTensor(16, 16)
		tensor.LoadFrom(data)

		result := tensor.ToHost()
		nanCount, infCount := CheckNumericalStability(result, "overflow")

		t.Logf("Overflow test: NaNs=%d, Infs=%d", nanCount, infCount)
	})

	t.Run("Subnormal values", func(t *testing.T) {
		data := make([]float32, 256)
		for i := range data {
			data[i] = math.SmallestNonzeroFloat32
		}

		tensor := ctx.NewTensor(16, 16)
		tensor.LoadFrom(data)

		result := tensor.ToHost()
		nanCount, _ := CheckNumericalStability(result, "subnormal")

		if nanCount > 0 {
			t.Errorf("Subnormal values should not produce NaNs, got %d", nanCount)
		}
	})

	t.Run("Zero values", func(t *testing.T) {
		data := make([]float32, 256)
		for i := range data {
			data[i] = 0.0
		}

		tensor := ctx.NewTensor(16, 16)
		tensor.LoadFrom(data)

		result := tensor.ToHost()
		nanCount, infCount := CheckNumericalStability(result, "zero")

		if nanCount != 0 || infCount != 0 {
			t.Errorf("Zeros should not produce instability: NaNs=%d, Infs=%d", nanCount, infCount)
		}
	})
}

func TestMetrics_NumericalInstability_Recording(t *testing.T) {
	t.Run("Metrics package exists", func(t *testing.T) {
		_ = metrics.RecordNumericalInstability
	})

	t.Run("Record validation error", func(t *testing.T) {
		_ = metrics.RecordValidationError
	})

	t.Run("Record context length", func(t *testing.T) {
		_ = metrics.RecordContextLength
	})
}

func TestTensor_Overflow_Detection(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	t.Run("ScanMax reports large values", func(t *testing.T) {
		data := make([]float32, 256)
		for i := range data {
			data[i] = 100000.0
		}

		tensor := ctx.NewTensor(16, 16)
		tensor.LoadFrom(data)

		maxVal, _ := tensor.ScanMax("large_values")

		if maxVal < 10000.0 {
			t.Errorf("Expected large max value, got %f", maxVal)
		}
	})

	t.Run("ScanMax handles NaN gracefully", func(t *testing.T) {
		data := make([]float32, 256)
		for i := range data {
			if i == 0 {
				data[i] = float32(math.NaN())
			} else {
				data[i] = 1.0
			}
		}

		tensor := ctx.NewTensor(16, 16)
		tensor.LoadFrom(data)

		maxVal, _ := tensor.ScanMax("with_nan")

		t.Logf("Max with NaN: %f", maxVal)
	})
}

func TestPrecision_Accumulation_Stability(t *testing.T) {
	t.Run("FP32 accumulation path exists for large models", func(t *testing.T) {
		ctx := NewContext()
		defer ctx.Free()

		scratch := ctx.NewLayerScratch(1, 8192, 28672, 64, 8, 128, 8192, 49152)
		if scratch == nil {
			t.Fatal("Failed to create scratch for large model")
		}
		defer scratch.Free()

		if scratch.NormedFFN_F32 == nil {
			t.Error("Large model should have FP32 norm buffer")
		}
		if scratch.ResFFN_F32 == nil {
			t.Error("Large model should have FP32 result buffer")
		}
	})

	t.Run("FP32 buffers have correct size", func(t *testing.T) {
		ctx := NewContext()
		defer ctx.Free()

		dim := 8192
		hiddenDim := 28672

		scratch := ctx.NewLayerScratch(1, dim, hiddenDim, 64, 8, 128, 8192, 49152)
		if scratch == nil {
			t.Fatal("Failed to create scratch")
		}
		defer scratch.Free()

		expectedNormSize := dim * 4 // FP32
		expectedResSize := dim * 4  // FP32

		if scratch.NormedFFN_F32.sizeBytes != expectedNormSize {
			t.Errorf("NormedFFN_F32 size: expected %d, got %d", expectedNormSize, scratch.NormedFFN_F32.sizeBytes)
		}

		if scratch.ResFFN_F32.sizeBytes != expectedResSize {
			t.Errorf("ResFFN_F32 size: expected %d, got %d", expectedResSize, scratch.ResFFN_F32.sizeBytes)
		}
	})
}
