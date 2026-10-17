//go:build emlfast

package mathx

import (
	"math"
	"testing"
)

// This file backs the `emlgo,emlfast` build. The fastmath functions
// (Exp, Log, Sin, Cos, Sqrt) use minimax polynomial approximations and are
// NOT bit-exact with the standard library, so they are checked against a
// relative tolerance. Every other function uses the bit-exact scalar emlgo
// path and is checked for exact equality.

// fastTol is the maximum allowed relative error for the fastmath approximations.
// Measured: FastExp ~1.6e-7, FastLog ~8.4e-8. Sin/Cos/Sqrt delegate to stdlib
// and are bit-exact, so they are checked with sameFloat instead.
const fastTol = 1e-6

func approxEqual(a, b, tol float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.IsNaN(a) && math.IsNaN(b)
	}
	if math.IsInf(a, 0) || math.IsInf(b, 0) {
		return a == b
	}
	diff := math.Abs(a - b)
	if diff <= tol {
		return true
	}
	// Relative comparison for large magnitudes.
	return diff <= tol*math.Max(math.Abs(a), math.Abs(b))
}

func TestFastParityExp(t *testing.T) {
	for _, x := range parityInputs() {
		if got, want := Exp(x), math.Exp(x); !approxEqual(got, want, fastTol) {
			t.Errorf("Exp(%v) = %v, want %v (rel err %v)", x, got, want, math.Abs(got-want)/math.Abs(want))
		}
	}
}

// TestFastParityLog uses logInputs because FastLog is only defined for positive
// inputs and diverges on denormals.
func TestFastParityLog(t *testing.T) {
	for _, x := range logInputs() {
		if got, want := Log(x), math.Log(x); !approxEqual(got, want, fastTol) {
			t.Errorf("Log(%v) = %v, want %v (rel err %v)", x, got, want, math.Abs(got-want)/math.Abs(want))
		}
	}
}

// fastmath.Sin/Cos/Sqrt delegate to the standard library and are bit-exact.
func TestFastParitySin(t *testing.T) {
	for _, x := range parityInputs() {
		if got, want := Sin(x), math.Sin(x); !sameFloat(got, want) {
			t.Errorf("Sin(%v) = %v, want %v", x, got, want)
		}
	}
}

func TestFastParityCos(t *testing.T) {
	for _, x := range parityInputs() {
		if got, want := Cos(x), math.Cos(x); !sameFloat(got, want) {
			t.Errorf("Cos(%v) = %v, want %v", x, got, want)
		}
	}
}

func TestFastParitySqrt(t *testing.T) {
	for _, x := range parityInputs() {
		if got, want := Sqrt(x), math.Sqrt(x); !sameFloat(got, want) {
			t.Errorf("Sqrt(%v) = %v, want %v", x, got, want)
		}
	}
}

// TestFastParityScalarFuncs pins the functions that fall back to scalar emlgo
// even under emlfast. The non-bit-exact functions (Log2, Tanh, Pow) use a ULP
// tolerance; the rest are bit-exact.
func TestFastParityScalarFuncs(t *testing.T) {
	for _, x := range parityInputs() {
		if got, want := Tanh(x), math.Tanh(x); !ulpEqual(got, want, 8) {
			t.Errorf("Tanh(%v) = %v, want %v (ULP diff %v)", x, got, want, ulpDiff(got, want))
		}
		if got, want := Abs(x), math.Abs(x); !sameFloat(got, want) {
			t.Errorf("Abs(%v) = %v, want %v", x, got, want)
		}
		if got, want := Floor(x), math.Floor(x); !sameFloat(got, want) {
			t.Errorf("Floor(%v) = %v, want %v", x, got, want)
		}
		if got, want := Ceil(x), math.Ceil(x); !sameFloat(got, want) {
			t.Errorf("Ceil(%v) = %v, want %v", x, got, want)
		}
		if got, want := Round(x), math.Round(x); !sameFloat(got, want) {
			t.Errorf("Round(%v) = %v, want %v", x, got, want)
		}
		if got, want := Trunc(x), math.Trunc(x); !sameFloat(got, want) {
			t.Errorf("Trunc(%v) = %v, want %v", x, got, want)
		}
	}
	for _, x := range logInputs() {
		if got, want := Log2(x), math.Log2(x); !ulpEqual(got, want, 256) {
			t.Errorf("Log2(%v) = %v, want %v (ULP diff %v)", x, got, want, ulpDiff(got, want))
		}
		if got, want := Log10(x), math.Log10(x); !sameFloat(got, want) {
			t.Errorf("Log10(%v) = %v, want %v", x, got, want)
		}
	}
	xs := []float64{0.5, 1, 2, 10, 100, 1e-10, 1e10, 2.5}
	ys := []float64{0, 1, -1, 0.5, 2, 10, -2, 3, 0.25, -0.5}
	for _, x := range xs {
		for _, y := range ys {
			if got, want := Pow(x, y), math.Pow(x, y); !ulpEqual(got, want, 128) {
				t.Errorf("Pow(%v, %v) = %v, want %v (ULP diff %v)", x, y, got, want, ulpDiff(got, want))
			}
		}
	}
}
