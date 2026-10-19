//go:build emlgo && !emlfast

package mathx

import (
	"github.com/emlgo/eml/pkg/arithmetic"
	"github.com/emlgo/eml/pkg/hyper"
	"github.com/emlgo/eml/pkg/logexp"
	"github.com/emlgo/eml/pkg/trig"
)

// This file is the emlgo backend. The scalar emlgo functions delegate to the
// Go standard library math package, so this build is bit-exact with the !emlgo
// build. The SIMD batch paths in emlgo's internal/eml package are not used
// here; they are opt-in for future per-op vectorization.

// Exp returns e**x, the base-e exponential of x.
func Exp(x float64) float64 { return logexp.Exp(x) }

// Log returns the natural logarithm of x.
func Log(x float64) float64 { return logexp.Log(x) }

// Log2 returns the binary logarithm of x.
func Log2(x float64) float64 { return arithmetic.LogBase2(x) }

// Log10 returns the decimal logarithm of x.
func Log10(x float64) float64 { return arithmetic.LogBase10(x) }

// Sqrt returns the square root of x.
func Sqrt(x float64) float64 { return arithmetic.Sqrt(x) }

// Pow returns x**y, the base-x exponential of y.
func Pow(x, y float64) float64 { return arithmetic.Pow(x, y) }

// Sin returns the sine of the radian argument x.
func Sin(x float64) float64 { return trig.Sin(x) }

// Cos returns the cosine of the radian argument x.
func Cos(x float64) float64 { return trig.Cos(x) }

// Tanh returns the hyperbolic tangent of x.
func Tanh(x float64) float64 { return hyper.Tanh(x) }

// Abs returns the absolute value of x.
func Abs(x float64) float64 { return arithmetic.Abs(x) }

// Floor returns the greatest integer value less than or equal to x.
func Floor(x float64) float64 { return arithmetic.Floor(x) }

// Ceil returns the least integer value greater than or equal to x.
func Ceil(x float64) float64 { return arithmetic.Ceil(x) }

// Round returns the nearest integer, rounding half away from zero.
func Round(x float64) float64 { return arithmetic.Round(x) }

// Trunc returns the integer value of x.
func Trunc(x float64) float64 { return arithmetic.Trunc(x) }
