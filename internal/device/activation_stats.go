package device

// activation_stats.go provides platform-agnostic activation analysis.
// This file is compiled into all builds and interacts with the platform-specific
// Tensor implementation through its public API (Rows, Cols, ToHost).

// ActivationStats contains comprehensive activation statistics
type ActivationStats struct {
	Max    float32
	Min    float32
	Mean   float32
	RMS    float32
	Zeros  int
	NaNs   int
	Infs   int
	Sample []float32 // First 16 values
}
