//go:build arm64 && !noasm

package simd

func init() {
	softmaxImpl = softmaxNEON
}

func softmaxNEON(x []float64) {
	// TODO: Implement NEON version
	softmaxFallback(x)
}
