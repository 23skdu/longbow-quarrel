//go:build darwin && metal

package engine

import (
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

// TestLogitRangeAudit tests that logit distribution is reasonable
// Addresses: nextsteps.md Item 4 - "Logit Range Audit: Inspect raw logit distribution; check for flatness or extreme values"
func TestLogitRangeAudit(t *testing.T) {
	// Test logit flatness detection
	tests := []struct {
		name          string
		logits        []float32
		expectFlat    bool
		expectExtreme bool
		expectNaN     bool
		maxVal        float32
		minVal        float32
		flatnessRatio float32
	}{
		{
			name:          "normal logits",
			logits:        []float32{-2.5, -1.2, 0.5, 3.1, 1.8, -0.3},
			expectFlat:    false,
			expectExtreme: false,
			maxVal:        3.1,
			minVal:        -2.5,
		},
		{
			name:          "flat logits - all same value",
			logits:        []float32{1.0, 1.0, 1.0, 1.0, 1.0},
			expectFlat:    true,
			expectExtreme: false,
			flatnessRatio: 1.0,
		},
		{
			name:          "flat logits - near zero variance",
			logits:        []float32{0.01, 0.011, 0.009, 0.01, 0.0105},
			expectFlat:    true, // Mean ~0.0102, RMS ~0.0102, variance near 0 - IS flat
			expectExtreme: false,
		},
		{
			name:          "flat logits - near zero centered",
			logits:        []float32{0.001, 0.0009, 0.0011, 0.001, 0.00105},
			expectFlat:    true, // Mean ~0.001, RMS ~0.001, both near zero
			expectExtreme: false,
		},
		{
			name:          "extreme values - NaN",
			logits:        []float32{1.0, float32(math.NaN()), 2.0, 3.0},
			expectNaN:     true,
			expectExtreme: true,
		},
		{
			name:          "extreme values - Inf",
			logits:        []float32{1.0, float32(math.Inf(1)), 2.0, 3.0},
			expectExtreme: true,
		},
		{
			name:          "extreme values - overflow",
			logits:        []float32{1e30, 1e31, 1e32, 1e33},
			expectExtreme: true,
		},
		{
			name:          "typical model output",
			logits:        []float32{-8.5, -4.2, 2.1, 5.3, 0.8, -1.1, -3.7, 1.4},
			expectFlat:    false,
			expectExtreme: false,
			maxVal:        5.3,
			minVal:        -8.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			audit := AuditLogitRange(tt.logits)

			if tt.expectNaN && !audit.HasNaN {
				t.Errorf("Expected NaN detection, but none found")
			}

			if tt.expectExtreme && !audit.HasExtremeValues {
				t.Errorf("Expected extreme value detection, but none found")
			}

			if tt.expectFlat && !audit.IsFlat {
				t.Errorf("Expected flat detection, but none found")
			}

			if !tt.expectFlat && audit.IsFlat {
				t.Errorf("Unexpected flat detection")
			}

			if tt.maxVal != 0 && audit.Max != tt.maxVal {
				t.Errorf("Expected max %v, got %v", tt.maxVal, audit.Max)
			}

			if tt.minVal != 0 && audit.Min != tt.minVal {
				t.Errorf("Expected min %v, got %v", tt.minVal, audit.Min)
			}

			// Record metrics for monitoring
			metrics.RecordLogitAudit(audit.Max, audit.Min, audit.Mean, audit.RMS, audit.HasNaN, audit.HasExtremeValues, audit.IsFlat)
		})
	}
}

// TestKVCacheAudit tests KV cache position logic
// Addresses: nextsteps.md Item 5 - "KV Cache Audit: Ensure CachePos logic doesn't cause overwrites or misindexing"
func TestKVCacheAudit(t *testing.T) {
	tests := []struct {
		name          string
		cacheSize     int // windowSize or seqLen
		positions     []int
		expectOverlap bool
		expectOOB     bool
	}{
		{
			name:          "normal sequential positions",
			cacheSize:     4096,
			positions:     []int{0, 1, 2, 3, 4, 5},
			expectOverlap: false,
			expectOOB:     false,
		},
		{
			name:          "wrapping at window boundary",
			cacheSize:     4096,
			positions:     []int{4094, 4095, 0, 1}, // wraps around
			expectOverlap: false,                   // by design in sliding window
			expectOOB:     false,
		},
		{
			name:          "duplicate position - potential overwrite",
			cacheSize:     4096,
			positions:     []int{0, 1, 1, 2}, // duplicate at position 1
			expectOverlap: true,
			expectOOB:     false,
		},
		{
			name:          "out of bounds position",
			cacheSize:     4096,
			positions:     []int{0, 1, 5000},
			expectOverlap: false,
			expectOOB:     true,
		},
		{
			name:          "sliding window behavior",
			cacheSize:     4096,
			positions:     []int{0, 1, 4095, 0, 1, 2}, // positions 0,1 appear twice (expected sliding behavior)
			expectOverlap: false,                      // sliding window intentionally overwrites
			expectOOB:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			audit := AuditKVCachePositions(tt.cacheSize, tt.positions)

			if tt.expectOverlap && !audit.HasOverlap {
				t.Errorf("Expected overlap detection, but none found")
			}

			if tt.expectOOB && !audit.HasOutOfBounds {
				t.Errorf("Expected out-of-bounds detection, but none found")
			}

			if !tt.expectOverlap && audit.HasOverlap && !audit.IsSlidingWindow {
				t.Errorf("Unexpected overlap detected for non-sliding window")
			}

			// Record metrics
			metrics.RecordKVCacheAudit(audit)
		})
	}
}

// TestScratchBufferSizing validates Scores buffer sizing
// Addresses: nextsteps.md Item 6 - "Scratch Buffer Sizing: Validate Scores buffer sizing (heads * seqLen * 4) and heap non-overlap"
func TestScratchBufferSizing(t *testing.T) {
	tests := []struct {
		name           string
		batch          int
		dim            int
		heads          int
		kvHeads        int
		headDim        int
		seqLen         int
		vocabSize      int
		expectedScores int // expected scores buffer size in elements
		expectValid    bool
	}{
		{
			name:           "Llama 3 8B config",
			batch:          1,
			dim:            4096,
			heads:          32,
			kvHeads:        8,
			headDim:        128,
			seqLen:         2048,
			vocabSize:      128256,
			expectedScores: 65536, // 32 * 2048, but will be 4096-aligned to 262144
			expectValid:    true,
		},
		{
			name:           "Mistral config",
			batch:          1,
			dim:            4096,
			heads:          32,
			kvHeads:        8,
			headDim:        128,
			seqLen:         4096, // Mistral sliding window
			vocabSize:      32000,
			expectedScores: 131072, // 32 * 4096, but will be 4096-aligned to 524288
			expectValid:    true,
		},
		{
			name:           "Small model config",
			batch:          1,
			dim:            2048,
			heads:          32,
			kvHeads:        8,
			headDim:        64,
			seqLen:         2048,
			vocabSize:      49152,
			expectedScores: 65536, // 32 * 2048, but will be 4096-aligned to 262144
			expectValid:    true,
		},
		{
			name:           "TinyLlama config",
			batch:          1,
			dim:            2048,
			heads:          32,
			kvHeads:        32, // MHA, not GQA
			headDim:        64,
			seqLen:         2048,
			vocabSize:      32000,
			expectedScores: 65536, // 32 * 2048, but will be 4096-aligned to 262144
			expectValid:    true,
		},
		{
			name:           "Q4K large model - ensure 32K minimum",
			batch:          1,
			dim:            8192,
			heads:          64,
			kvHeads:        8,
			headDim:        128,
			seqLen:         512,
			vocabSize:      128256,
			expectedScores: 32768, // 64 * 512, but will be at least 32K (131072 after 4096-align)
			expectValid:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			audit := AuditScratchBufferSizing(tt.batch, tt.dim, tt.heads, tt.kvHeads, tt.headDim, tt.seqLen, tt.vocabSize)

			if tt.expectValid && !audit.IsValid {
				t.Errorf("Expected valid buffer sizing, got: %v", audit)
			}

			// Verify scores buffer size after alignment
			// ExpectedScoresSize is in bytes (ExpectedScores * 4), ExpectedScores is element count
			expectedBytes := tt.expectedScores * 4
			if audit.ExpectedScoresSize != expectedBytes {
				t.Errorf("Expected scores size (bytes) %d, got %d", expectedBytes, audit.ExpectedScoresSize)
			}

			// Verify headDim calculation
			expectedHeadDim := tt.dim / tt.heads
			if audit.HeadDim != expectedHeadDim {
				t.Errorf("Expected headDim %d, got %d", expectedHeadDim, audit.HeadDim)
			}

			// Verify GQA ratio
			expectedGQARatio := tt.heads / tt.kvHeads
			if audit.GQARatio != expectedGQARatio {
				t.Errorf("Expected GQA ratio %d, got %d", expectedGQARatio, audit.GQARatio)
			}

			// Verify 4096 alignment
			if audit.ActualScoresSize%4096 != 0 {
				t.Errorf("Scores size %d not aligned to 4096", audit.ActualScoresSize)
			}

			// Record metrics
			metrics.RecordBufferSizingAudit(audit)
		})
	}
}

// TestDequantizationAccuracy tests CPU-side dequantization
// Addresses: nextsteps.md Item 7 - "Dequantization Accuracy: Verify CPU-side Q6_K dequantization matches reference outputs"
func TestDequantizationAccuracy(t *testing.T) {
	tests := []struct {
		name        string
		quantType   device.DataType
		inputData   []byte
		numElements int
		maxRelError float64
		expectPass  bool
	}{
		{
			name:        "Q6_K simple pattern",
			quantType:   device.DataTypeQ6K,
			inputData:   generateSimpleQ6KPattern(256),
			numElements: 256,
			maxRelError: 0.01, // 1% relative error tolerance
			expectPass:  true,
		},
		{
			name:        "Q4_K simple pattern",
			quantType:   device.DataTypeQ4K,
			inputData:   generateSimpleQ4KPattern(256),
			numElements: 256,
			maxRelError: 0.02, // 2% relative error tolerance for Q4_K
			expectPass:  true,
		},
		{
			name:        "F16 direct - should be exact",
			quantType:   device.DataTypeF16,
			inputData:   generateF16Pattern(128),
			numElements: 128,
			maxRelError: 0.0,
			expectPass:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			audit := AuditDequantizationAccuracy(tt.quantType, tt.inputData, tt.numElements)

			if tt.expectPass && !audit.Passed {
				t.Errorf("Expected dequantization to pass, but failed: maxRelError=%v, maxAbsError=%v",
					audit.MaxRelError, audit.MaxAbsError)
			}

			if audit.MaxRelError > tt.maxRelError {
				t.Errorf("Relative error %v exceeds tolerance %v", audit.MaxRelError, tt.maxRelError)
			}

			// Record metrics
			metrics.RecordDequantizationAudit(audit)
		})
	}
}

// TestWeightPaddingAlignment tests weight padding and alignment
// Addresses: nextsteps.md Item 8 - "Weight Padding/Alignment: Investigate token_embd.weight zero-padding and alignment offsets"
func TestWeightPaddingAlignment(t *testing.T) {
	tests := []struct {
		name          string
		tensorName    string
		reportedRows  int
		reportedCols  int
		actualRows    int
		actualCols    int
		ggufType      int
		expectPadding bool
		expectAligned bool
		paddingBytes  int
	}{
		{
			name:          "token_embd with Q4_K quantization",
			tensorName:    "token_embd.weight",
			reportedRows:  49152,
			reportedCols:  4096,
			actualRows:    49152 + 256, // Q4_K adds padding for alignment
			actualCols:    4096,
			ggufType:      12, // Q4_K
			expectPadding: true,
			expectAligned: true,
			paddingBytes:  256 * 4096, // padding from alignment
		},
		{
			name:          "token_embd F16 exact",
			tensorName:    "token_embd.weight",
			reportedRows:  32000,
			reportedCols:  4096,
			actualRows:    32000,
			actualCols:    4096,
			ggufType:      1, // F16
			expectPadding: false,
			expectAligned: true,
			paddingBytes:  0,
		},
		{
			name:          "ffn_down with Q4_K alignment",
			tensorName:    "blk.0.ffn_down.weight",
			reportedRows:  14336,
			reportedCols:  4096,
			actualRows:    14336,
			actualCols:    4096,
			ggufType:      12,    // Q4_K
			expectPadding: false, // Q4_K uses block-based quantization
			expectAligned: true,
			paddingBytes:  0,
		},
		{
			name:          "Q4_0 with alignment requirement",
			tensorName:    "output.weight",
			reportedRows:  128256,
			reportedCols:  4096,
			actualRows:    128256,
			actualCols:    4096,
			ggufType:      2,     // Q4_0
			expectPadding: false, // Q4_0 pads internally
			expectAligned: true,
			paddingBytes:  0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			audit := AuditWeightPadding(tt.tensorName, tt.reportedRows, tt.reportedCols,
				tt.actualRows, tt.actualCols, tt.ggufType)

			if tt.expectPadding && !audit.HasPadding {
				t.Errorf("Expected padding detection, but none found")
			}

			if tt.expectAligned && !audit.IsAligned {
				t.Errorf("Expected alignment check failed")
			}

			if tt.paddingBytes > 0 && audit.PaddingBytes != tt.paddingBytes {
				t.Errorf("Expected %d padding bytes, got %d", tt.paddingBytes, audit.PaddingBytes)
			}

			// Record metrics
			metrics.RecordWeightAlignmentAudit(audit)
		})
	}
}

// TestSoftmaxAttentionMasking tests softmax masking behavior
// Addresses: nextsteps.md Item 9 - "Softmax Attention Masking: Ensure softmax_f16 strictly masks tokens beyond pos"
func TestSoftmaxAttentionMasking(t *testing.T) {
	tests := []struct {
		name         string
		seqLen       int
		pos          int
		windowSize   int
		scores       [][]float32
		expectMasked []bool // which positions should be masked at each pos
		expectStrict bool
	}{
		{
			name:       "normal attention - no masking",
			seqLen:     4096,
			pos:        10,
			windowSize: 0, // no sliding window
			scores: [][]float32{
				{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0}, // pos 0-10
			},
			expectMasked: []bool{false, false, false, false, false, false, false, false, false, false, false},
			expectStrict: true,
		},
		{
			name:       "sliding window 4096 - all within window",
			seqLen:     4096,
			pos:        100,
			windowSize: 4096,
			scores:     generateSlidingWindowScores(100, 4096),
			expectMasked: func() []bool {
				m := make([]bool, 4096)
				for i := 0; i <= 100; i++ {
					m[i] = false // within window
				}
				return m
			}(),
			expectStrict: true,
		},
		{
			name:       "sliding window - positions beyond window masked",
			seqLen:     4096,
			pos:        4100,
			windowSize: 4096,
			scores:     generateSlidingWindowScores(4100, 4096),
			expectMasked: func() []bool {
				m := make([]bool, 4096)
				for i := 0; i < 4; i++ {
					m[i] = true // positions 0-3 are beyond sliding window (pos 4100 - 4096 = 4)
				}
				// For positions 4-4095, they are within the window
				return m
			}(),
			expectStrict: true,
		},
		{
			name:       "causal masking - current token not masked",
			seqLen:     4096,
			pos:        9, // Query is at position 9
			windowSize: 0,
			scores: [][]float32{
				{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, -100000.0}, // position 10 is future token
			},
			expectMasked: []bool{false, false, false, false, false, false, false, false, false, false, true},
			expectStrict: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			audit := AuditSoftmaxMasking(tt.seqLen, tt.pos, tt.windowSize, tt.scores)

			if tt.expectStrict && !audit.IsStrict {
				t.Errorf("Expected strict masking, but found unmasked positions")
			}

			// Verify expected masked positions
			for i, expected := range tt.expectMasked {
				if i >= len(audit.MaskedPositions) {
					continue
				}
				if audit.MaskedPositions[i] != expected {
					t.Errorf("Position %d: expected masked=%v, got %v", i, expected, audit.MaskedPositions[i])
				}
			}

			// Verify mask value is sufficiently negative
			if audit.MaskValue > -1e4 {
				t.Errorf("Mask value %v should be very negative (< -10000)", audit.MaskValue)
			}

			// Record metrics
			metrics.RecordSoftmaxMaskingAudit(audit)
		})
	}
}

// TestHeadDimensionLogic tests head dimension handling in kernels
// Addresses: nextsteps.md Item 10 - "Head Dimension Logic: Confirm headDim=128 handling in kernels is correct for threadgroups"
func TestHeadDimensionLogic(t *testing.T) {
	tests := []struct {
		name            string
		headDim         int
		numHeads        int
		seqLen          int
		expectedThreads int // expected threads per threadgroup
		threadgroupOK   bool
		alignmentOK     bool
	}{
		{
			name:            "Llama 3 standard config",
			headDim:         128,
			numHeads:        32,
			seqLen:          2048,
			expectedThreads: 128, // should match headDim for optimal coalescing
			threadgroupOK:   true,
			alignmentOK:     true,
		},
		{
			name:            "Mistral standard config",
			headDim:         128,
			numHeads:        32,
			seqLen:          4096,
			expectedThreads: 128,
			threadgroupOK:   true,
			alignmentOK:     true,
		},
		{
			name:            "Small model config",
			headDim:         64,
			numHeads:        32,
			seqLen:          2048,
			expectedThreads: 64,
			threadgroupOK:   true,
			alignmentOK:     true,
		},
		{
			name:            "TinyLlama MHA config",
			headDim:         64,
			numHeads:        32,
			seqLen:          2048,
			expectedThreads: 64,
			threadgroupOK:   true,
			alignmentOK:     true,
		},
		{
			name:            "Large model config",
			headDim:         128,
			numHeads:        64,
			seqLen:          4096,
			expectedThreads: 128,
			threadgroupOK:   true,
			alignmentOK:     true,
		},
		{
			name:            "Non-power-of-2 headDim - problematic",
			headDim:         96,
			numHeads:        32,
			seqLen:          2048,
			expectedThreads: 96,    // may cause threadgroup issues
			threadgroupOK:   false, // Not power of 2
			alignmentOK:     false,
		},
		{
			name:            "Non-power-of-2 headDim - evenly divisible",
			headDim:         96,
			numHeads:        64, // 64 heads * 64 seqLen = 4096 total threads
			seqLen:          64,
			expectedThreads: 96,
			threadgroupOK:   false, // Not power of 2
			alignmentOK:     false,
		},
		{
			name:            "Non-power-of-2 headDim - evenly divisible",
			headDim:         96,
			numHeads:        64,
			seqLen:          64,
			expectedThreads: 96,
			threadgroupOK:   false, // Not power of 2
			alignmentOK:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			audit := AuditHeadDimensionLogic(tt.headDim, tt.numHeads, tt.seqLen)

			if audit.ThreadgroupSize != tt.expectedThreads {
				t.Errorf("Expected threadgroup size %d, got %d", tt.expectedThreads, audit.ThreadgroupSize)
			}

			// Verify threadgroup size is power of 2 (optimal for Metal)
			isPowerOf2 := audit.HeadDim&(audit.HeadDim-1) == 0
			if tt.threadgroupOK && !isPowerOf2 {
				t.Errorf("headDim %d is not power of 2, may cause threadgroup issues", audit.HeadDim)
			}

			// For power-of-2 head dimensions, verify threadgroup divides evenly
			if isPowerOf2 && audit.TotalThreads%audit.ThreadgroupSize != 0 {
				t.Errorf("Total threads %d not evenly divisible by threadgroup size %d",
					audit.TotalThreads, audit.ThreadgroupSize)
			}

			// Record metrics
			metrics.RecordHeadDimensionAudit(audit)
		})
	}
}

// Helper functions for generating test data

func generateSimpleQ6KPattern(n int) []byte {
	// Generate minimal valid Q6_K data
	data := make([]byte, n*210/256*210) // approximate size
	for i := range data {
		data[i] = byte(i)
	}
	return data
}

func generateSimpleQ4KPattern(n int) []byte {
	// Generate minimal valid Q4_K data
	data := make([]byte, n*144/256*144) // approximate size
	for i := range data {
		data[i] = byte(i)
	}
	return data
}

func generateF16Pattern(n int) []byte {
	// Generate F16 values as bytes
	data := make([]byte, n*2)
	for i := 0; i < n; i++ {
		// Simple pattern: 1.0, 2.0, 3.0, ...
		val := float32(i + 1)
		bits := math.Float32bits(val)
		data[i*2] = byte(bits)
		data[i*2+1] = byte(bits >> 8)
	}
	return data
}

func generateSlidingWindowScores(pos, windowSize int) [][]float32 {
	// Generate scores for sliding window attention
	// Size should match seqLen (windowSize or seqLen from test)
	scores := make([]float32, windowSize)
	for i := 0; i < windowSize; i++ {
		scores[i] = float32(i % 10) // simple pattern
	}
	return [][]float32{scores}
}

// Additional comprehensive tests

// TestActivationFlowAnalysis tests layer-by-layer activation flow
// Addresses: "Analyze Layer 0 -> Layer 31 activation flow" from nextsteps.md
func TestActivationFlowAnalysis(t *testing.T) {
	numLayers := 32
	traces := make([]device.ActivationStats, numLayers)

	// Simulate normal activation flow
	for i := 0; i < numLayers; i++ {
		traces[i] = device.ActivationStats{
			Max:   float32(5.0 + float32(i)*0.1), // slight increase per layer
			Min:   float32(-5.0 - float32(i)*0.1),
			Mean:  0.0,
			RMS:   float32(2.0 + float32(i)*0.05),
			Zeros: 0,
			NaNs:  0,
			Infs:  0,
		}
	}

	audit := AuditActivationFlow(traces, numLayers)

	if !audit.IsHealthy {
		t.Errorf("Expected healthy activation flow, but found issues: %v", audit)
	}

	// Verify no collapsed layers
	if len(audit.CollapsedLayers) > 0 {
		t.Errorf("Found collapsed layers: %v", audit.CollapsedLayers)
	}

	// Verify no saturated layers
	if len(audit.SaturatedLayers) > 0 {
		t.Errorf("Found saturated layers: %v", audit.SaturatedLayers)
	}

	// Verify monotonic flow (no sudden jumps)
	for i := 1; i < len(traces); i++ {
		delta := traces[i].Max - traces[i-1].Max
		if math.Abs(float64(delta)) > 10.0 {
			t.Errorf("Large activation jump at layer %d: delta=%v", i, delta)
		}
	}

	metrics.RecordActivationFlowAudit(len(audit.CollapsedLayers), len(audit.SaturatedLayers), 0)
}

// TestNaNPropagationDetection detects NaN propagation issues
// Addresses: Current critical problem - NaN starting around Layer 23
func TestNaNPropagationDetection(t *testing.T) {
	// Simulate the observed issue: NaNs starting at Layer 23
	numLayers := 32
	traces := make([]device.ActivationStats, numLayers)

	for i := 0; i < numLayers; i++ {
		if i >= 23 {
			// NaN propagation starts here
			traces[i] = device.ActivationStats{
				Max:  float32(math.NaN()),
				Min:  float32(math.NaN()),
				NaNs: 4096, // All values are NaN
			}
		} else {
			traces[i] = device.ActivationStats{
				Max:  float32(5.0),
				Min:  float32(-5.0),
				NaNs: 0,
			}
		}
	}

	audit := AuditNaNPropagation(traces)

	if !audit.HasNaN {
		t.Errorf("Expected NaN detection, but none found")
	}

	if audit.NaNLayerStart != 23 {
		t.Errorf("Expected NaN to start at layer 23, got layer %d", audit.NaNLayerStart)
	}

	if audit.NaNLayerEnd != 31 {
		t.Errorf("Expected NaN to continue through layer 31, got layer %d", audit.NaNLayerEnd)
	}

	metrics.RecordNaNPropagationAudit(audit.NaNLayerStart, audit.NaNLayerEnd, audit.TotalNaNCount, audit.Pattern)
}

// TestRoPEDeviationCheck tests RoPE precision
// Addresses: "RoPE precision issue: Tests reveal 0.045 deviation for Mistral 1M theta"
func TestRoPEDeviationCheck(t *testing.T) {
	tests := []struct {
		name           string
		theta          float32
		expectedMaxDev float32
		actualMaxDev   float32
		expectPass     bool
	}{
		{
			name:           "Llama 2 10K theta",
			theta:          10000.0,
			expectedMaxDev: 0.001,
			actualMaxDev:   0.0005,
			expectPass:     true,
		},
		{
			name:           "Mistral 1M theta - known issue",
			theta:          1000000.0,
			expectedMaxDev: 0.05, // Relaxed tolerance for known issue
			actualMaxDev:   0.045,
			expectPass:     true,
		},
		{
			name:           "Mistral 1M theta - failing",
			theta:          1000000.0,
			expectedMaxDev: 0.001, // Strict tolerance
			actualMaxDev:   0.045,
			expectPass:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			audit := AuditRoPEDeviation(tt.theta, tt.expectedMaxDev, tt.actualMaxDev)

			if tt.expectPass && !audit.Passed {
				t.Errorf("Expected RoPE deviation check to pass, actual=%v, expected=%v",
					tt.actualMaxDev, tt.expectedMaxDev)
			}

			if !tt.expectPass && audit.Passed {
				t.Errorf("Expected RoPE deviation check to fail, but passed")
			}

			metrics.RecordRoPEDeviationAudit(audit.ActualMaxDev, audit.DeviationRatio, audit.Passed)
		})
	}
}
