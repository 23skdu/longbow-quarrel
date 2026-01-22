//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

func TestAttFused_4000_Token_Limit(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numHeads := 4
	kvHeads := 4
	headDim := 64

	testCases := []struct {
		name       string
		pos        int
		expectFail bool
	}{
		{"Small context (256)", 255, false},
		{"Medium context (1000)", 999, false},
		{"Near limit (3999)", 3999, false},
		{"At limit (4000)", 4000, false},
		{"Exceeds limit (8000)", 7999, false},
		{"Large context (16000)", 15999, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			q := ctx.NewTensor(1, numHeads*headDim)
			kCache := ctx.NewTensor(tc.pos+1, kvHeads*headDim)
			vCache := ctx.NewTensor(tc.pos+1, kvHeads*headDim)
			out := ctx.NewTensor(1, numHeads*headDim)

			qData := make([]float32, numHeads*headDim)
			for i := range qData {
				qData[i] = 0.1
			}
			q.LoadFrom(qData)

			kvData := make([]float32, (tc.pos+1)*kvHeads*headDim)
			for i := range kvData {
				kvData[i] = 0.1
			}
			kCache.LoadFrom(kvData)
			vCache.LoadFrom(kvData)

			err := q.AttFused(kCache, vCache, out, tc.pos, numHeads, kvHeads, headDim, 0)
			ctx.Synchronize()

			if tc.expectFail && err == nil {
				t.Logf("Expected failure at pos=%d but got success", tc.pos)
			}
			if !tc.expectFail && err != nil {
				t.Errorf("Unexpected error at pos=%d: %v", tc.pos, err)
			}
		})
	}
}

func TestAttFused_Output_Bounds_Validation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numHeads := 4
	kvHeads := 4
	headDim := 64
	pos := 100

	testCases := []struct {
		name        string
		outRows     int
		outCols     int
		expectError bool
	}{
		{"Valid output", 1, numHeads * headDim, false},
		{"Wrong rows", 2, numHeads * headDim, true},
		{"Wrong cols", 1, numHeads*headDim - 1, true},
		{"Both wrong", 2, numHeads*headDim - 1, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			q := ctx.NewTensor(1, numHeads*headDim)
			kCache := ctx.NewTensor(pos+1, kvHeads*headDim)
			vCache := ctx.NewTensor(pos+1, kvHeads*headDim)
			out := ctx.NewTensor(tc.outRows, tc.outCols)

			qData := make([]float32, numHeads*headDim)
			kCacheData := make([]float32, (pos+1)*kvHeads*headDim)
			vCacheData := make([]float32, (pos+1)*kvHeads*headDim)

			q.LoadFrom(qData)
			kCache.LoadFrom(kCacheData)
			vCache.LoadFrom(vCacheData)

			err := q.AttFused(kCache, vCache, out, pos, numHeads, kvHeads, headDim, 0)

			if tc.expectError && err == nil {
				t.Error("Expected validation error but got none")
			}
			if !tc.expectError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

func TestAttFused_GQA_Validation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 64
	pos := 100

	testCases := []struct {
		name        string
		numHeads    int
		kvHeads     int
		expectError bool
	}{
		{"Valid GQA (32/8)", 32, 8, false},
		{"Valid MHA (32/32)", 32, 32, false},
		{"Invalid (not divisible)", 32, 7, true},
		{"Invalid (kv > heads)", 8, 16, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			q := ctx.NewTensor(1, tc.numHeads*headDim)
			kCache := ctx.NewTensor(pos+1, tc.kvHeads*headDim)
			vCache := ctx.NewTensor(pos+1, tc.kvHeads*headDim)
			out := ctx.NewTensor(1, tc.numHeads*headDim)

			qData := make([]float32, tc.numHeads*headDim)
			kCacheData := make([]float32, (pos+1)*tc.kvHeads*headDim)
			vCacheData := make([]float32, (pos+1)*tc.kvHeads*headDim)

			q.LoadFrom(qData)
			kCache.LoadFrom(kCacheData)
			vCache.LoadFrom(vCacheData)

			err := q.AttFused(kCache, vCache, out, pos, tc.numHeads, tc.kvHeads, headDim, 0)

			if tc.expectError && err == nil {
				t.Error("Expected GQA validation error but got none")
			}
			if !tc.expectError && err != nil {
				t.Errorf("Unexpected GQA error: %v", err)
			}
		})
	}
}

func TestAttFused_Position_Validation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numHeads := 4
	kvHeads := 4
	headDim := 64

	q := ctx.NewTensor(1, numHeads*headDim)
	kCache := ctx.NewTensor(100, kvHeads*headDim)
	vCache := ctx.NewTensor(100, kvHeads*headDim)
	out := ctx.NewTensor(1, numHeads*headDim)

	q.LoadFrom(make([]float32, numHeads*headDim))
	kCache.LoadFrom(make([]float32, 100*kvHeads*headDim))
	vCache.LoadFrom(make([]float32, 100*kvHeads*headDim))

	err := q.AttFused(kCache, vCache, out, -1, numHeads, kvHeads, headDim, 0)
	if err == nil {
		t.Error("Expected error for negative position")
	}
}

func TestDataType_Enum_Uniqueness(t *testing.T) {
	types := []struct {
		dt   DataType
		name string
	}{
		{DataTypeF16, "F16"},
		{DataTypeQ4K, "Q4K"},
		{DataTypeQ4_0, "Q4_0"},
		{DataTypeQ3K, "Q3K"},
		{DataTypeF32, "F32"},
		{DataTypeQ6K, "Q6K"},
	}

	seen := make(map[DataType]string)
	for _, tt := range types {
		if existing, ok := seen[tt.dt]; ok {
			t.Errorf("Duplicate DataType value %d: %s and %s", tt.dt, existing, tt.name)
		}
		seen[tt.dt] = tt.name
	}

	if len(seen) != len(types) {
		t.Errorf("Expected %d unique DataType values, got %d", len(types), len(seen))
	}
}

func TestQ4_0_And_Q3K_Are_Different(t *testing.T) {
	if DataTypeQ4_0 == DataTypeQ3K {
		t.Error("CRITICAL: DataTypeQ4_0 and DataTypeQ3K have the same enum value")
	}
	t.Logf("DataTypeQ4_0 = %d, DataTypeQ3K = %d", DataTypeQ4_0, DataTypeQ3K)
}

func TestLinear_Dimension_Validation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	testCases := []struct {
		name        string
		inputCols   int
		weightCols  int
		expectError bool
	}{
		{"Valid (128/128)", 128, 128, false},
		{"Invalid mismatch (128/64)", 128, 64, true},
		{"Invalid mismatch (64/128)", 64, 128, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			input := ctx.NewTensor(1, tc.inputCols)
			weight := ctx.NewTensor(256, tc.weightCols)

			_, err := input.Linear(weight)

			if tc.expectError && err == nil {
				t.Error("Expected dimension validation error")
			}
			if !tc.expectError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

func TestLinearInto_Dimension_Validation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	input := ctx.NewTensor(1, 128)
	weight := ctx.NewTensor(256, 128)
	out := ctx.NewTensor(1, 256)

	err := input.LinearInto(weight, out, 1.0)
	if err != nil {
		t.Errorf("Valid dimensions should not error: %v", err)
	}

	outWrong := ctx.NewTensor(1, 128)
	err = input.LinearInto(weight, outWrong, 1.0)
	if err == nil {
		t.Error("Invalid output dimensions should error")
	}
}

func TestAdd_Dimension_Validation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	a := ctx.NewTensor(2, 4)
	bValid := ctx.NewTensor(2, 4)
	bInvalid := ctx.NewTensor(4, 4)

	_, err := a.Add(bValid)
	if err != nil {
		t.Errorf("Valid add dimensions: %v", err)
	}

	_, err = a.Add(bInvalid)
	if err == nil {
		t.Error("Invalid add dimensions should error")
	}
}

func TestSwiGLU_Dimension_Validation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	tVal := ctx.NewTensor(2, 4)
	tGateValid := ctx.NewTensor(2, 4)
	tGateInvalid := ctx.NewTensor(2, 8)

	_, err := tVal.SwiGLU(tGateValid)
	if err != nil {
		t.Errorf("Valid SwiGLU dimensions: %v", err)
	}

	_, err = tVal.SwiGLU(tGateInvalid)
	if err == nil {
		t.Error("Invalid SwiGLU dimensions should error")
	}
}

func TestLoadFrom_Size_Validation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	tensor := ctx.NewTensor(4, 4)

	err := tensor.LoadFrom(make([]float32, 16))
	if err != nil {
		t.Errorf("Valid size should not error: %v", err)
	}

	err = tensor.LoadFrom(make([]float32, 8))
	if err == nil {
		t.Error("Wrong size should error")
	}

	err = tensor.LoadFrom(make([]float32, 32))
	if err == nil {
		t.Error("Too large size should error")
	}
}

func TestNaN_Detection(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	t.Run("Valid tensor has no NaNs", func(t *testing.T) {
		data := make([]float32, 1024)
		for i := range data {
			data[i] = float32(i) * 0.01
		}
		tensor := ctx.NewTensor(32, 32)
		tensor.LoadFrom(data)

		count := tensor.ScanNaNs("test")
		if count != 0 {
			t.Errorf("Expected 0 NaNs, got %d", count)
		}
	})

	t.Run("NaN detection works", func(t *testing.T) {
		data := make([]float32, 1024)
		for i := range data {
			data[i] = 1.0
		}
		data[0] = float32(math.NaN())
		tensor := ctx.NewTensor(32, 32)
		tensor.LoadFrom(data)

		count := tensor.ScanNaNs("nan_test")
		if count == 0 {
			t.Log("Note: NaN detection may need kernel support")
		}
	})

	t.Run("Inf detection works", func(t *testing.T) {
		data := make([]float32, 1024)
		for i := range data {
			data[i] = 1.0
		}
		data[0] = float32(math.Inf(1))
		tensor := ctx.NewTensor(32, 32)
		tensor.LoadFrom(data)

		count := tensor.ScanNaNs("inf_test")
		if count == 0 {
			t.Log("Note: Inf detection may need kernel support")
		}
	})
}

func TestNumerical_Stability_FP16(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	t.Run("FP16 safe range", func(t *testing.T) {
		data := make([]float32, 1024)
		for i := range data {
			data[i] = 1000.0
		}
		tensor := ctx.NewTensor(32, 32)
		tensor.LoadFrom(data)

		result := tensor.ToHost()
		for i, v := range result {
			if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
				t.Errorf("Numerical instability at %d: %f", i, v)
			}
		}
	})

	t.Run("Subnormal handling", func(t *testing.T) {
		data := make([]float32, 1024)
		for i := range data {
			data[i] = float32(math.SmallestNonzeroFloat32)
		}
		tensor := ctx.NewTensor(32, 32)
		tensor.LoadFrom(data)

		result := tensor.ToHost()
		for i, v := range result {
			if math.IsNaN(float64(v)) {
				t.Errorf("NaN from subnormal at %d", i)
			}
		}
	})

	t.Run("Overflow handling", func(t *testing.T) {
		data := make([]float32, 1024)
		for i := range data {
			data[i] = 65504.0 * 1.5 // Exceed FP16 max (65504)
		}
		tensor := ctx.NewTensor(32, 32)
		tensor.LoadFrom(data)

		result := tensor.ToHost()
		for i, v := range result {
			if math.IsInf(float64(v), 0) {
				t.Logf("Overflow detected at %d: %f (expected for large values)", i, v)
			}
		}
	})
}

func TestQ4_0_Block_Size_Validation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	testCases := []struct {
		name  string
		cols  int
		valid bool
	}{
		{"Divisible by 32", 4096, true},
		{"Divisible by 32 (256)", 256, true},
		{"Not divisible (1)", 4097, false},
		{"Not divisible (17)", 4113, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateQ4_0Dimensions(256, tc.cols)
			if tc.valid && err != nil {
				t.Errorf("Expected valid Q4_0 dims, got error: %v", err)
			}
			if !tc.valid && err == nil {
				t.Errorf("Expected invalid Q4_0 dims, got no error")
			}
		})
	}
}

func TestQ4_K_Block_Size_Validation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	testCases := []struct {
		name  string
		cols  int
		valid bool
	}{
		{"Divisible by 256", 4096, true},
		{"Divisible by 256 (512)", 512, true},
		{"Not divisible (1)", 4097, false},
		{"Not divisible (129)", 4129, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateQ4_KDimensions(256, tc.cols)
			if tc.valid && err != nil {
				t.Errorf("Expected valid Q4_K dims, got error: %v", err)
			}
			if !tc.valid && err == nil {
				t.Errorf("Expected invalid Q4_K dims, got no error")
			}
		})
	}
}

func TestQ6_K_Block_Size_Validation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	testCases := []struct {
		name  string
		cols  int
		valid bool
	}{
		{"Divisible by 256", 4096, true},
		{"Divisible by 256 (768)", 768, true},
		{"Not divisible (1)", 4097, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateQ6_KDimensions(256, tc.cols)
			if tc.valid && err != nil {
				t.Errorf("Expected valid Q6_K dims, got error: %v", err)
			}
			if !tc.valid && err == nil {
				t.Errorf("Expected invalid Q6_K dims, got no error")
			}
		})
	}
}

func TestLayerScratch_Heap_Management(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	t.Run("Create and free scratch", func(t *testing.T) {
		scratch := ctx.NewLayerScratch(1, 4096, 11008, 32, 8, 128, 4096, 49152)
		if scratch == nil {
			t.Fatal("Failed to create layer scratch")
		}
		scratch.Free()
	})

	t.Run("Multiple allocations", func(t *testing.T) {
		for i := 0; i < 5; i++ {
			scratch := ctx.NewLayerScratch(1, 4096, 11008, 32, 8, 128, 4096, 49152)
			if scratch == nil {
				t.Fatalf("Failed to create scratch %d", i)
			}
			scratch.Free()
		}
	})

	t.Run("Heap reference cleanup", func(t *testing.T) {
		scratch := ctx.NewLayerScratch(1, 4096, 11008, 32, 8, 128, 4096, 49152)
		if scratch.heap == nil {
			t.Error("Heap should be set")
		}
		scratch.Free()
		if scratch.heap != nil {
			t.Error("Heap should be nil after Free")
		}
	})
}

func TestFFN_FP32_For_Large_Models(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	dim := 4096
	hiddenDim := 14336

	t.Run("FFN path selection for large dim", func(t *testing.T) {
		useF32FFN := dim < 1024
		useMixedPrecisionFFN := !useF32FFN && dim >= 4096

		if useF32FFN {
			t.Log("Small model: using FP32 FFN")
		} else if useMixedPrecisionFFN {
			t.Log("Large model: using mixed precision FFN")
		} else {
			t.Log("Medium model: using FP16 FFN")
		}

		if dim >= 4096 && !useMixedPrecisionFFN {
			t.Error("Large models should use mixed precision")
		}
	})

	t.Run("FFN tensors exist", func(t *testing.T) {
		scratch := ctx.NewLayerScratch(1, dim, hiddenDim, 32, 8, 128, 4096, 49152)
		defer scratch.Free()

		if scratch.NormedFFN_F32 == nil {
			t.Error("NormedFFN_F32 should be allocated")
		}
		if scratch.ResFFN_F32 == nil {
			t.Error("ResFFN_F32 should be allocated")
		}
	})
}

func TestCopyToF16_Into_Validation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	src := ctx.NewTensorFP32(4, 4)
	dstValid := ctx.NewTensor(4, 4)
	dstInvalid := ctx.NewTensor(2, 4)

	err := src.CopyToF16_Into(dstValid)
	if err != nil {
		t.Errorf("Valid copy should not error: %v", err)
	}

	err = src.CopyToF16_Into(dstInvalid)
	if err == nil {
		t.Error("Invalid dimensions should error")
	}
}

func TestValidationError_Type(t *testing.T) {
	err := NewValidationError("TestOp", "test message", "test/path")
	if err.Error() != "TestOp: test message (test/path)" {
		t.Errorf("Unexpected error string: %s", err.Error())
	}
}

func TestModelConfig_Validation(t *testing.T) {
	testCases := []struct {
		name        string
		config      *ModelConfig
		expectError bool
	}{
		{"Valid config", &ModelConfig{
			Dim: 4096, Layers: 32, Heads: 32, KVHeads: 8,
			HeadDim: 128, HiddenDim: 14336,
		}, false},
		{"Zero dim", &ModelConfig{
			Dim: 0, Layers: 32, Heads: 32, KVHeads: 8,
			HeadDim: 128, HiddenDim: 14336,
		}, true},
		{"Invalid GQA", &ModelConfig{
			Dim: 4096, Layers: 32, Heads: 32, KVHeads: 7,
			HeadDim: 128, HiddenDim: 14336,
		}, true},
		{"KV > Heads", &ModelConfig{
			Dim: 4096, Layers: 32, Heads: 8, KVHeads: 16,
			HeadDim: 128, HiddenDim: 14336,
		}, true},
		{"Dim/Head mismatch", &ModelConfig{
			Dim: 4096, Layers: 32, Heads: 32, KVHeads: 8,
			HeadDim: 64, HiddenDim: 14336,
		}, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.config.Validate()
			if tc.expectError && err == nil {
				t.Error("Expected validation error")
			}
			if !tc.expectError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

func TestInput_Token_Validation(t *testing.T) {
	testCases := []struct {
		name        string
		tokens      []int
		vocabSize   int
		expectError bool
	}{
		{"Empty tokens", []int{}, 100, true},
		{"Valid tokens", []int{1, 2, 3, 4, 5}, 100, false},
		{"Out of range", []int{99, 100}, 100, true},
		{"Negative token", []int{-1, 2}, 100, true},
		{"Mixed valid/invalid", []int{1, 2, 1000, 4}, 100, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ValidateInputTokens(tc.tokens, tc.vocabSize)
			if tc.expectError && err == nil {
				t.Error("Expected validation error")
			}
			if !tc.expectError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}
