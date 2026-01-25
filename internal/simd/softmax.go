package simd

import "math"

var softmaxImpl func(x []float64)

func Softmax(x []float64) {
	softmaxImpl(x)
}

func init() {
	softmaxImpl = softmaxFallback
}

func softmaxFallback(x []float64) {
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
