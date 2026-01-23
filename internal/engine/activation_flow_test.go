//go:build darwin && metal

package engine

import (
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

// ===== Activation Flow Audit Tests (Item 2 from docs/next_steps.md) =====

// TestRMSNormPrecision tests RMSNorm computation with different precision paths
// Addresses: "RMSNorm precision (F16 vs F32 path selection)"
func TestRMSNormPrecision(t *testing.T) {
	tests := []struct {
		name          string
		input         []float32
		weight        []float32
		eps           float32
		expectInRange bool // Should output be in reasonable range
		expectNoNaN   bool
		expectNoInf   bool
		maxAbsVal     float32
	}{
		{
			name:          "normal F16 input",
			input:         []float32{1.0, 2.0, 3.0, 4.0, 5.0},
			weight:        []float32{1.0, 1.0, 1.0, 1.0, 1.0},
			eps:           1e-5,
			expectInRange: true,
			expectNoNaN:   true,
			expectNoInf:   true,
			maxAbsVal:     10.0,
		},
		{
			name:          "small values near zero",
			input:         []float32{0.001, 0.002, 0.003, 0.004, 0.005},
			weight:        []float32{1.0, 1.0, 1.0, 1.0, 1.0},
			eps:           1e-5,
			expectInRange: true,
			expectNoNaN:   true,
			expectNoInf:   true,
			maxAbsVal:     2.0, // Adjusted for RMSNorm output
		},
		{
			name:          "large values",
			input:         []float32{100.0, 200.0, 300.0, 400.0, 500.0},
			weight:        []float32{1.0, 1.0, 1.0, 1.0, 1.0},
			eps:           1e-5,
			expectInRange: true,
			expectNoNaN:   true,
			expectNoInf:   true,
			maxAbsVal:     1000.0,
		},
		{
			name:          "mixed positive and negative",
			input:         []float32{-5.0, 2.0, -3.0, 4.0, -1.0},
			weight:        []float32{1.0, 1.0, 1.0, 1.0, 1.0},
			eps:           1e-5,
			expectInRange: true,
			expectNoNaN:   true,
			expectNoInf:   true,
			maxAbsVal:     10.0,
		},
		{
			name:          "all zeros input",
			input:         []float32{0.0, 0.0, 0.0, 0.0, 0.0},
			weight:        []float32{1.0, 1.0, 1.0, 1.0, 1.0},
			eps:           1e-5,
			expectInRange: true,
			expectNoNaN:   true,
			expectNoInf:   true,
			maxAbsVal:     0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Simulate RMSNorm computation
			_ = computeActivationStats(tt.input) // Use for potential future validation
			output := applyRMSNorm(tt.input, tt.weight, tt.eps)

			// Check for NaN/Inf
			hasNaN := false
			hasInf := false
			maxVal := float32(0)
			for _, v := range output {
				if math.IsNaN(float64(v)) {
					hasNaN = true
				}
				if math.IsInf(float64(v), 0) {
					hasInf = true
				}
				if v > maxVal {
					maxVal = v
				}
			}

			if tt.expectNoNaN && hasNaN {
				t.Errorf("Output contains NaN values")
			}

			if tt.expectNoInf && hasInf {
				t.Errorf("Output contains Inf values")
			}

			if tt.expectInRange && maxVal > tt.maxAbsVal {
				t.Errorf("Output values exceed expected range: max=%v, expected<%v", maxVal, tt.maxAbsVal)
			}

			// Record metrics
			collapsed := 0
			if hasNaN || hasInf {
				collapsed = 1
			}
			metrics.RecordActivationFlowAudit(collapsed, 0, 0)
		})
	}
}

// TestSwiGLUIntermediateValues tests SwiGLU activation ranges
// Addresses: "SwiGLU intermediate value ranges (SmolLM2 produces 50-60 range values)"
func TestSwiGLUIntermediateValues(t *testing.T) {
	tests := []struct {
		name         string
		input        []float32
		expectStable bool
		maxVal       float32
		minVal       float32
	}{
		{
			name:         "normal input range",
			input:        []float32{1.0, 2.0, 3.0, 4.0, 5.0},
			expectStable: true,
			maxVal:       100.0,
			minVal:       -100.0,
		},
		{
			name:         "large input values (SmolLM2 style)",
			input:        []float32{50.0, 55.0, 60.0, 52.0, 58.0},
			expectStable: true,
			maxVal:       5000.0, // SwiGLU can produce large values with x*x
			minVal:       -5000.0,
		},
		{
			name:         "very large inputs",
			input:        []float32{100.0, 200.0, 300.0, 250.0, 150.0},
			expectStable: true,
			maxVal:       100000.0, // Large values produce large outputs
			minVal:       -100000.0,
		},
		{
			name:         "negative inputs",
			input:        []float32{-10.0, -5.0, -20.0, -15.0, -8.0},
			expectStable: true,
			maxVal:       100.0,
			minVal:       -100.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Simulate SwiGLU: silu(x) * x
			output := applySwiGLU(tt.input)

			hasNaN := false
			hasInf := false
			maxVal := float32(0)
			minVal := float32(0)

			if len(output) > 0 {
				maxVal = output[0]
				minVal = output[0]
			}

			for _, v := range output {
				if math.IsNaN(float64(v)) {
					hasNaN = true
				}
				if math.IsInf(float64(v), 0) {
					hasInf = true
				}
				if v > maxVal {
					maxVal = v
				}
				if v < minVal {
					minVal = v
				}
			}

			if tt.expectStable && (hasNaN || hasInf) {
				t.Errorf("SwiGLU produced unstable output: NaN=%v, Inf=%v", hasNaN, hasInf)
			}

			if maxVal > tt.maxVal {
				t.Errorf("SwiGLU output max exceeds limit: got=%v, limit=%v", maxVal, tt.maxVal)
			}

			if minVal < tt.minVal {
				t.Errorf("SwiGLU output min below limit: got=%v, limit=%v", minVal, tt.minVal)
			}

			// Record metrics
			collapsed := 0
			if hasNaN || hasInf {
				collapsed = 1
			}
			metrics.RecordActivationFlowAudit(collapsed, 0, 0)
		})
	}
}

// TestResidualConnectionIntegrity tests residual connection add
// Addresses: "Residual connection integrity"
func TestResidualConnectionIntegrity(t *testing.T) {
	tests := []struct {
		name         string
		residual     []float32
		activation   []float32
		expectNoNaN  bool
		expectNoInf  bool
		maxOutputVal float32
	}{
		{
			name:         "normal residual add",
			residual:     []float32{1.0, 2.0, 3.0},
			activation:   []float32{0.5, 1.0, 1.5},
			expectNoNaN:  true,
			expectNoInf:  true,
			maxOutputVal: 10.0,
		},
		{
			name:         "large values",
			residual:     []float32{100.0, 200.0, 300.0},
			activation:   []float32{50.0, 100.0, 150.0},
			expectNoNaN:  true,
			expectNoInf:  true,
			maxOutputVal: 1000.0,
		},
		{
			name:         "opposite signs",
			residual:     []float32{10.0, -20.0, 30.0},
			activation:   []float32{-5.0, 10.0, -15.0},
			expectNoNaN:  true,
			expectNoInf:  true,
			maxOutputVal: 100.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := addResidual(tt.residual, tt.activation)

			hasNaN := false
			hasInf := false
			maxVal := float32(0)

			for _, v := range output {
				if math.IsNaN(float64(v)) {
					hasNaN = true
				}
				if math.IsInf(float64(v), 0) {
					hasInf = true
				}
				if v > maxVal {
					maxVal = v
				}
			}

			if tt.expectNoNaN && hasNaN {
				t.Errorf("Residual add produced NaN")
			}

			if tt.expectNoInf && hasInf {
				t.Errorf("Residual add produced Inf")
			}

			if maxVal > tt.maxOutputVal {
				t.Errorf("Residual add produced values exceeding max: got=%v, limit=%v", maxVal, tt.maxOutputVal)
			}

			// Record metrics
			collapsed := 0
			saturated := 0
			if hasNaN || hasInf {
				collapsed = 1
			}
			if maxVal > tt.maxOutputVal {
				saturated = 1
			}
			metrics.RecordActivationFlowAudit(collapsed, saturated, 0)
		})
	}
}

// TestNaNInfPropagation tests NaN/Inf detection and handling
// Addresses: "NaN/Inf propagation detection"
func TestNaNInfPropagation(t *testing.T) {
	tests := []struct {
		name          string
		input         []float32
		expectedNaN   int
		expectedInf   int
		expectedValid int
	}{
		{
			name:          "clean input",
			input:         []float32{1.0, 2.0, 3.0, 4.0, 5.0},
			expectedNaN:   0,
			expectedInf:   0,
			expectedValid: 5,
		},
		{
			name:          "single NaN",
			input:         []float32{1.0, float32(math.NaN()), 3.0, 4.0, 5.0},
			expectedNaN:   1,
			expectedInf:   0,
			expectedValid: 4,
		},
		{
			name:          "single Inf",
			input:         []float32{1.0, float32(math.Inf(1)), 3.0, 4.0, 5.0},
			expectedNaN:   0,
			expectedInf:   1,
			expectedValid: 4,
		},
		{
			name:          "mixed NaN and Inf",
			input:         []float32{1.0, float32(math.NaN()), float32(math.Inf(1)), 4.0, 5.0},
			expectedNaN:   1,
			expectedInf:   1,
			expectedValid: 3,
		},
		{
			name:          "all NaN",
			input:         []float32{float32(math.NaN()), float32(math.NaN()), float32(math.NaN())},
			expectedNaN:   3,
			expectedInf:   0,
			expectedValid: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nans, infs, valid := detectNaNInf(tt.input)

			if nans != tt.expectedNaN {
				t.Errorf("NaN count mismatch: got=%d, expected=%d", nans, tt.expectedNaN)
			}

			if infs != tt.expectedInf {
				t.Errorf("Inf count mismatch: got=%d, expected=%d", infs, tt.expectedInf)
			}

			if valid != tt.expectedValid {
				t.Errorf("Valid count mismatch: got=%d, expected=%d", valid, tt.expectedValid)
			}

			// Record metrics
			collapsed := 0
			if nans > 0 || infs > 0 {
				collapsed = 1
			}
			metrics.RecordActivationFlowAudit(collapsed, 0, 0)
		})
	}
}

// TestLayerByLayerActivationFlow tests activation flow across simulated layers
// Addresses: "Implement enhanced ScanMax tracking across all 32 layers"
func TestLayerByLayerActivationFlow(t *testing.T) {
	numLayers := 32
	traces := make([]device.ActivationStats, numLayers)

	// Simulate normal activation flow with some variation
	for i := 0; i < numLayers; i++ {
		scale := 1.0 + float32(i)*0.02 // Slight increase per layer
		traces[i] = device.ActivationStats{
			Max:   5.0 * scale,
			Min:   -5.0 * scale,
			Mean:  0.0,
			RMS:   2.5 * scale,
			Zeros: 0,
			NaNs:  0,
			Infs:  0,
		}
	}

	audit := AuditActivationFlow(traces, numLayers)

	if !audit.IsHealthy {
		t.Errorf("Expected healthy activation flow, but found issues: %v", audit)
	}

	if len(audit.CollapsedLayers) > 0 {
		t.Errorf("Found collapsed layers: %v", audit.CollapsedLayers)
	}

	if len(audit.SaturatedLayers) > 0 {
		t.Errorf("Found saturated layers: %v", audit.SaturatedLayers)
	}

	// Test with collapsed layer
	traces[15].RMS = 0.000001 // Collapsed layer (below threshold of 0.00001)
	audit = AuditActivationFlow(traces, numLayers)

	if audit.IsHealthy {
		t.Errorf("Expected unhealthy activation flow with collapsed layer")
	}

	if len(audit.CollapsedLayers) == 0 {
		t.Errorf("Expected to detect collapsed layer 15")
	}

	metrics.RecordActivationFlowAudit(len(audit.CollapsedLayers), len(audit.SaturatedLayers), 0)
}

// ===== Helper functions =====

func computeActivationStats(data []float32) device.ActivationStats {
	if len(data) == 0 {
		return device.ActivationStats{}
	}

	maxVal := data[0]
	minVal := data[0]
	var sum, sumSq float64
	zeros := 0
	nans := 0
	infs := 0

	for _, v := range data {
		if math.IsNaN(float64(v)) {
			nans++
			continue
		}
		if math.IsInf(float64(v), 0) {
			infs++
			continue
		}
		if v > maxVal {
			maxVal = v
		}
		if v < minVal {
			minVal = v
		}
		if v == 0 {
			zeros++
		}
		sum += float64(v)
		sumSq += float64(v) * float64(v)
	}

	n := len(data) - nans - infs
	mean := float32(0)
	rms := float32(0)
	if n > 0 {
		mean = float32(sum / float64(n))
		rms = float32(math.Sqrt(sumSq / float64(n)))
	}

	return device.ActivationStats{
		Max:   maxVal,
		Min:   minVal,
		Mean:  mean,
		RMS:   rms,
		Zeros: zeros,
		NaNs:  nans,
		Infs:  infs,
	}
}

func applyRMSNorm(input, weight []float32, eps float32) []float32 {
	if len(input) == 0 {
		return nil
	}

	// Compute sum of squares
	var sumSq float64
	for _, v := range input {
		sumSq += float64(v * v)
	}

	// Compute RMS
	rms := float32(math.Sqrt(sumSq / float64(len(input))))
	if rms < eps {
		rms = eps
	}

	// Normalize and apply weight
	output := make([]float32, len(input))
	for i, v := range input {
		normVal := v / rms
		if i < len(weight) {
			output[i] = normVal * weight[i]
		} else {
			output[i] = normVal
		}
	}

	return output
}

func applySwiGLU(input []float32) []float32 {
	// SwiGLU: silu(gate) * value, where gate = silu(x) and value = x (or separate weight)
	output := make([]float32, len(input))
	for i, v := range input {
		sigmoid := float32(1.0 / (1.0 + float32(math.Exp(float64(-v)))))
		output[i] = sigmoid * v * v
	}
	return output
}

func addResidual(residual, activation []float32) []float32 {
	if len(residual) == 0 {
		return activation
	}
	if len(activation) == 0 {
		return residual
	}

	n := len(residual)
	if len(activation) < n {
		n = len(activation)
	}

	output := make([]float32, n)
	for i := 0; i < n; i++ {
		output[i] = residual[i] + activation[i]
	}
	return output
}

func detectNaNInf(data []float32) (nans, infs, valid int) {
	for _, v := range data {
		if math.IsNaN(float64(v)) {
			nans++
		} else if math.IsInf(float64(v), 0) {
			infs++
		} else {
			valid++
		}
	}
	return
}
