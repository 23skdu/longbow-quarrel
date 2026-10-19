//go:build !(amd64 && cgo) && !(arm64 && cgo)

package simd

// polarQuantAccelerated is the no-op accelerator for builds without a SIMD
// kernel. It reports that no acceleration is available so the caller falls
// back to the canonical reference implementation.
func polarQuantAccelerated(input []float32, rotationMatrix []float32, n, bits int) ([]int8, float32, []float32, bool) {
	return nil, 0, nil, false
}

// qjlTransformAccelerated is the no-op accelerator for builds without a SIMD
// kernel.
func qjlTransformAccelerated(residual []float32, signMatrix []float32, rows, cols int) ([]int8, float32, bool) {
	return nil, 0, false
}
