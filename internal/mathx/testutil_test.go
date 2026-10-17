package mathx

import "math"

// parityInputs is a deterministic sweep covering edge values, denormals, and
// large/small magnitudes. It is shared by every parity sub-test and has no
// build tag so it is available in all backend configurations.
func parityInputs() []float64 {
	return []float64{
		0, 1, -1, 0.5, -0.5, 2, -2, 10, -10, 100, -100,
		1e-300, -1e-300, 1e300, -1e300,
		math.MaxFloat64, -math.MaxFloat64,
		math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64,
		math.Pi, -math.Pi,
		0.1, 0.2, 0.3, 1.5, 2.5, 3.14159, 0.0001, 12345.6789,
	}
}

// positiveInputs is a sweep of strictly positive inputs, for the logarithm
// functions whose domain edge (x = 0) differs between emlgo and the standard
// library (emlgo returns NaN where stdlib returns -Inf).
func positiveInputs() []float64 {
	return []float64{
		1e-300, 1e-10, 0.0001, 0.001, 0.1, 0.5, 1, 1.5, 2, 10, 100,
		12345.6789, 1e10, 1e100, 1e300, math.MaxFloat64,
		math.Pi, math.SmallestNonzeroFloat64,
	}
}

// logInputs is positiveInputs without denormals. emlgo's Log is bit-exact with
// math.Log over the normal range but diverges on denormals (e.g. 5e-324), so
// those are excluded from the bit-exact comparison.
func logInputs() []float64 {
	var out []float64
	for _, x := range positiveInputs() {
		if x >= 1e-300 {
			out = append(out, x)
		}
	}
	return out
}

// sameFloat compares two float64 values for bit-exact equality, treating two
// NaNs as equal (NaN != NaN would otherwise fail every NaN-producing input).
func sameFloat(a, b float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.IsNaN(a) && math.IsNaN(b)
	}
	return a == b
}

// ulpDiff returns the number of representable float64 values between a and b.
// It returns +Inf when exactly one of a, b is NaN.
func ulpDiff(a, b float64) float64 {
	if math.IsNaN(a) || math.IsNaN(b) {
		if math.IsNaN(a) && math.IsNaN(b) {
			return 0
		}
		return math.Inf(1)
	}
	if a == b {
		return 0
	}
	ab, bb := math.Float64bits(a), math.Float64bits(b)
	if ab > bb {
		return float64(ab - bb)
	}
	return float64(bb - ab)
}

// ulpEqual reports whether a and b are within maxUlp representable float64
// values of each other. It is used for the emlgo functions that are not
// bit-exact with the standard library (Tanh, Pow, Log2).
func ulpEqual(a, b, maxUlp float64) bool {
	return ulpDiff(a, b) <= maxUlp
}
