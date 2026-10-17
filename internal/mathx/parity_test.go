//go:build !emlfast

package mathx

import (
	"math"
	"testing"
)

func TestParityExp(t *testing.T) {
	for _, x := range parityInputs() {
		if got, want := Exp(x), math.Exp(x); !sameFloat(got, want) {
			t.Errorf("Exp(%v) = %v, want %v", x, got, want)
		}
	}
}

// TestParityLog uses logInputs (positive, non-denormal) because emlgo's Log is
// bit-exact with math.Log over the normal range but diverges on denormals.
func TestParityLog(t *testing.T) {
	for _, x := range logInputs() {
		if got, want := Log(x), math.Log(x); !sameFloat(got, want) {
			t.Errorf("Log(%v) = %v, want %v", x, got, want)
		}
	}
}

// TestParityLog2 uses a ULP tolerance because emlgo's LogBase2 is not bit-exact
// with math.Log2 (measured max ~173 ULP over the positive domain) and returns
// NaN at x = 0 where math.Log2 returns -Inf, so only positive inputs are used.
func TestParityLog2(t *testing.T) {
	for _, x := range positiveInputs() {
		if got, want := Log2(x), math.Log2(x); !ulpEqual(got, want, 256) {
			t.Errorf("Log2(%v) = %v, want %v (ULP diff %v)", x, got, want, ulpDiff(got, want))
		}
	}
}

// TestParityLog10 uses logInputs because emlgo's LogBase10 returns NaN at x = 0
// where math.Log10 returns -Inf, and diverges on denormals. Over the normal
// positive domain it is bit-exact.
func TestParityLog10(t *testing.T) {
	for _, x := range logInputs() {
		if got, want := Log10(x), math.Log10(x); !sameFloat(got, want) {
			t.Errorf("Log10(%v) = %v, want %v", x, got, want)
		}
	}
}

func TestParitySqrt(t *testing.T) {
	for _, x := range parityInputs() {
		if got, want := Sqrt(x), math.Sqrt(x); !sameFloat(got, want) {
			t.Errorf("Sqrt(%v) = %v, want %v", x, got, want)
		}
	}
}

// TestParityPow uses a ULP tolerance because emlgo's Pow is not bit-exact with
// math.Pow (measured max ~90 ULP over this sweep). Only positive bases are used
// so the sweep stays on real-valued results.
func TestParityPow(t *testing.T) {
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

func TestParitySin(t *testing.T) {
	for _, x := range parityInputs() {
		if got, want := Sin(x), math.Sin(x); !sameFloat(got, want) {
			t.Errorf("Sin(%v) = %v, want %v", x, got, want)
		}
	}
}

func TestParityCos(t *testing.T) {
	for _, x := range parityInputs() {
		if got, want := Cos(x), math.Cos(x); !sameFloat(got, want) {
			t.Errorf("Cos(%v) = %v, want %v", x, got, want)
		}
	}
}

// TestParityTanh uses a ULP tolerance because emlgo's Tanh is not bit-exact
// with math.Tanh (measured max ~2 ULP).
func TestParityTanh(t *testing.T) {
	for _, x := range parityInputs() {
		if got, want := Tanh(x), math.Tanh(x); !ulpEqual(got, want, 8) {
			t.Errorf("Tanh(%v) = %v, want %v (ULP diff %v)", x, got, want, ulpDiff(got, want))
		}
	}
}

func TestParityAbs(t *testing.T) {
	for _, x := range parityInputs() {
		if got, want := Abs(x), math.Abs(x); !sameFloat(got, want) {
			t.Errorf("Abs(%v) = %v, want %v", x, got, want)
		}
	}
}

func TestParityMax(t *testing.T) {
	xs := []float64{0, 1, -1, 0.5, math.NaN(), math.Inf(1), math.Inf(-1), math.MaxFloat64}
	ys := []float64{0, -1, 2, -0.5, math.NaN(), math.Inf(-1), math.Inf(1), -math.MaxFloat64}
	for _, x := range xs {
		for _, y := range ys {
			if got, want := Max(x, y), math.Max(x, y); !sameFloat(got, want) {
				t.Errorf("Max(%v, %v) = %v, want %v", x, y, got, want)
			}
		}
	}
}

func TestParityMin(t *testing.T) {
	xs := []float64{0, 1, -1, 0.5, math.NaN(), math.Inf(1), math.Inf(-1), math.MaxFloat64}
	ys := []float64{0, -1, 2, -0.5, math.NaN(), math.Inf(-1), math.Inf(1), -math.MaxFloat64}
	for _, x := range xs {
		for _, y := range ys {
			if got, want := Min(x, y), math.Min(x, y); !sameFloat(got, want) {
				t.Errorf("Min(%v, %v) = %v, want %v", x, y, got, want)
			}
		}
	}
}

func TestParityFloor(t *testing.T) {
	for _, x := range parityInputs() {
		if got, want := Floor(x), math.Floor(x); !sameFloat(got, want) {
			t.Errorf("Floor(%v) = %v, want %v", x, got, want)
		}
	}
}

func TestParityCeil(t *testing.T) {
	for _, x := range parityInputs() {
		if got, want := Ceil(x), math.Ceil(x); !sameFloat(got, want) {
			t.Errorf("Ceil(%v) = %v, want %v", x, got, want)
		}
	}
}

func TestParityRound(t *testing.T) {
	for _, x := range parityInputs() {
		if got, want := Round(x), math.Round(x); !sameFloat(got, want) {
			t.Errorf("Round(%v) = %v, want %v", x, got, want)
		}
	}
}

func TestParityTrunc(t *testing.T) {
	for _, x := range parityInputs() {
		if got, want := Trunc(x), math.Trunc(x); !sameFloat(got, want) {
			t.Errorf("Trunc(%v) = %v, want %v", x, got, want)
		}
	}
}

// TestParityControlFlow pins the always-stdlib helpers so a regression in the
// shared mathx.go file fails loudly in every build configuration.
func TestParityControlFlow(t *testing.T) {
	if !IsNaN(NaN()) {
		t.Error("IsNaN(NaN()) = false, want true")
	}
	if !IsInf(Inf(1), 1) {
		t.Error("IsInf(Inf(1), 1) = false, want true")
	}
	if IsInf(1, 0) {
		t.Error("IsInf(1, 0) = true, want false")
	}
	if got, want := Float32bits(1.5), math.Float32bits(1.5); got != want {
		t.Errorf("Float32bits(1.5) = %v, want %v", got, want)
	}
	if got, want := Float32frombits(0x3fc00000), math.Float32frombits(0x3fc00000); got != want {
		t.Errorf("Float32frombits = %v, want %v", got, want)
	}
	if got, want := Float64bits(1.5), math.Float64bits(1.5); got != want {
		t.Errorf("Float64bits(1.5) = %v, want %v", got, want)
	}
	if got, want := Float64frombits(0x3ff8000000000000), math.Float64frombits(0x3ff8000000000000); got != want {
		t.Errorf("Float64frombits = %v, want %v", got, want)
	}
	if got, want := Copysign(-1.5, 1), math.Copysign(-1.5, 1); got != want {
		t.Errorf("Copysign = %v, want %v", got, want)
	}
	if MaxInt != math.MaxInt {
		t.Errorf("MaxInt = %v, want %v", MaxInt, math.MaxInt)
	}
	if MaxFloat32 != math.MaxFloat32 {
		t.Errorf("MaxFloat32 = %v, want %v", MaxFloat32, math.MaxFloat32)
	}
	if MaxFloat64 != math.MaxFloat64 {
		t.Errorf("MaxFloat64 = %v, want %v", MaxFloat64, math.MaxFloat64)
	}
	if SmallestNonzeroFloat32 != math.SmallestNonzeroFloat32 {
		t.Errorf("SmallestNonzeroFloat32 = %v, want %v", SmallestNonzeroFloat32, math.SmallestNonzeroFloat32)
	}
	if Pi != math.Pi {
		t.Errorf("Pi = %v, want %v", Pi, math.Pi)
	}
}
