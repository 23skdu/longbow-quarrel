//go:build !amd64 && !arm64

package simd

import "math"

func softmax(x []float64) {
	if len(x) == 0 {
		return
	}
	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}

	sum := 0.0
	for i := range x {
		x[i] = math.Exp(x[i] - max)
		sum += x[i]
	}

	for i := range x {
		x[i] /= sum
	}
}
