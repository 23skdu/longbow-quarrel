//go:build darwin && metal

package device

import (
	"errors"
	"fmt"
	"math"
	"sync/atomic"
)

type ModelConfig struct {
	Dim        int
	HiddenDim  int
	Layers     int
	Heads      int
	KVHeads    int
	HeadDim    int
	VocabSize  int
	SeqLen     int
	Eps        float32
	RopeTheta  float32
	WindowSize int
}

func ValidateInputTokens(tokens []int, vocabSize int) ([]int, error) {
	if len(tokens) == 0 {
		return nil, errors.New("empty input tokens")
	}

	validated := make([]int, len(tokens))
	for i, token := range tokens {
		if token < 0 || token >= vocabSize {
			return nil, fmt.Errorf("input token %d at position %d is out of vocab range [0, %d)", token, i, vocabSize)
		}
		validated[i] = token
	}
	return validated, nil
}

func (c *ModelConfig) Validate() error {
	if c.Dim <= 0 {
		return errors.New("invalid model dimension: must be positive")
	}
	if c.Layers <= 0 {
		return errors.New("invalid layer count: must be positive")
	}
	if c.Heads <= 0 {
		return errors.New("invalid attention head count: must be positive")
	}
	if c.KVHeads <= 0 {
		return errors.New("invalid KV head count: must be positive")
	}
	if c.Heads%c.KVHeads != 0 {
		return errors.New("KVHeads must evenly divide Heads for Grouped Query Attention")
	}
	if c.HeadDim <= 0 {
		return errors.New("invalid head dimension: must be positive")
	}
	if c.Dim != c.Heads*c.HeadDim {
		return fmt.Errorf("dimension mismatch: Dim=%d, Heads=%d, HeadDim=%d (expected %d)",
			c.Dim, c.Heads, c.HeadDim, c.Heads*c.HeadDim)
	}
	if c.HiddenDim <= 0 {
		return errors.New("invalid hidden dimension: must be positive")
	}
	return nil
}

func GetKernelDurationSum() float64 {
	return 0
}

func GetGPUMemoryAllocated() int64 {
	return atomic.LoadInt64(&allocatedBytes)
}

func ValidateTensorDimensions(aRows, aCols, bRows, bCols int) error {
	if aCols != bRows {
		return fmt.Errorf("matrix dimension mismatch: A[%d,%d] * B[%d,%d] invalid",
			aRows, aCols, bRows, bCols)
	}
	return nil
}

func ValidateLinearDimensions(inputCols, weightCols int) error {
	if inputCols != weightCols {
		return fmt.Errorf("linear dimension mismatch: input cols=%d != weight cols=%d",
			inputCols, weightCols)
	}
	return nil
}

func ValidateAddDimensions(aRows, aCols, bRows, bCols int) error {
	if aRows != bRows || aCols != bCols {
		return fmt.Errorf("add dimension mismatch: A[%d,%d] != B[%d,%d]",
			aRows, aCols, bRows, bCols)
	}
	return nil
}

type NaNInfo struct {
	Count     int
	Positions []int
	Values    []float32
	HasInf    bool
	InfCount  int
}

func (n *NaNInfo) HasNaN() bool {
	return n.Count > 0
}

func (n *NaNInfo) IsValid() bool {
	return n.Count == 0 && !n.HasInf
}

func CheckNumericalStability(data []float32, name string) (nanCount, infCount int) {
	for _, v := range data {
		if math.IsNaN(float64(v)) {
			nanCount++
		}
		if math.IsInf(float64(v), 0) {
			infCount++
		}
	}
	return
}

func DetectNaN(data []float32, maxPositions int) *NaNInfo {
	info := &NaNInfo{}
	for i, v := range data {
		if math.IsNaN(float64(v)) {
			info.Count++
			if len(info.Positions) < maxPositions {
				info.Positions = append(info.Positions, i)
				info.Values = append(info.Values, v)
			}
		}
		if math.IsInf(float64(v), 0) {
			info.HasInf = true
			info.InfCount++
		}
	}
	return info
}

func (t *Tensor) ScanForNaN(name string, maxReport int) *NaNInfo {
	data := t.ToHost()
	return DetectNaN(data, maxReport)
}

func ValidateAndReport(name string, data []float32, maxNaNAllowed int) (*NaNInfo, error) {
	info := DetectNaN(data, 10)
	if info.Count > maxNaNAllowed {
		return info, fmt.Errorf("%s: too many NaNs (%d > %d allowed), first positions: %v",
			name, info.Count, maxNaNAllowed, info.Positions)
	}
	return info, nil
}

func HasAnyNaN(data []float32) bool {
	for _, v := range data {
		if math.IsNaN(float64(v)) {
			return true
		}
	}
	return false
}

func HasAnyInf(data []float32) bool {
	for _, v := range data {
		if math.IsInf(float64(v), 0) {
			return true
		}
	}
	return false
}

func IsValid(data []float32) bool {
	return !HasAnyNaN(data) && !HasAnyInf(data)
}

func Float32Max(data []float32) float32 {
	if len(data) == 0 {
		return 0
	}
	max := data[0]
	for _, v := range data {
		if v > max {
			max = v
		}
	}
	return max
}

func Float32Min(data []float32) float32 {
	if len(data) == 0 {
		return 0
	}
	min := data[0]
	for _, v := range data {
		if v < min {
			min = v
		}
	}
	return min
}

func ValidateAttentionOutput(output *Tensor, numHeads, headDim int) error {
	expectedRows := 1
	expectedCols := numHeads * headDim

	if output.Rows() != expectedRows {
		return fmt.Errorf("attention output rows mismatch: expected %d, got %d",
			expectedRows, output.Rows())
	}
	if output.Cols() != expectedCols {
		return fmt.Errorf("attention output cols mismatch: expected %d, got %d",
			expectedCols, output.Cols())
	}
	return nil
}

func (t *Tensor) ValidateForOperation(op string, expectedRows, expectedCols int) error {
	if t.Rows() != expectedRows {
		return fmt.Errorf("%s: tensor rows mismatch: expected %d, got %d",
			op, expectedRows, t.Rows())
	}
	if t.Cols() != expectedCols {
		return fmt.Errorf("%s: tensor cols mismatch: expected %d, got %d",
			op, expectedCols, t.Cols())
	}
	return nil
}

type ValidationError struct {
	Op   string
	Msg  string
	Path string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("%s: %s (%s)", e.Op, e.Msg, e.Path)
}

func NewValidationError(op, msg, path string) *ValidationError {
	return &ValidationError{Op: op, Msg: msg, Path: path}
}

func ValidateQ4_0Dimensions(rows, cols int) error {
	if cols%32 != 0 {
		return fmt.Errorf("Q4_0 requires cols divisible by 32, got cols=%d", cols)
	}
	return nil
}

func ValidateQ4_KDimensions(rows, cols int) error {
	if cols%256 != 0 {
		return fmt.Errorf("Q4_K requires cols divisible by 256, got cols=%d", cols)
	}
	return nil
}

func ValidateQ6_KDimensions(rows, cols int) error {
	if cols%256 != 0 {
		return fmt.Errorf("Q6_K requires cols divisible by 256, got cols=%d", cols)
	}
	return nil
}
