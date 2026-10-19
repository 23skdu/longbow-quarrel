//go:build emlgo && emlfast

package mathx

import (
	"github.com/emlgo/eml/pkg/arithmetic"
	"github.com/emlgo/eml/pkg/fastmath"
	"github.com/emlgo/eml/pkg/hyper"
)

// This file is the emlgo fastmath backend. It swaps in emlgo's minimax
// polynomial approximations where available. These are NOT bit-exact with the
// standard library and may shift logits and therefore sampling; they exist to
// measure the speed/accuracy trade-off. Functions without a fastmath
// equivalent fall back to the bit-exact scalar emlgo path.

// Exp returns e**x, the base-e exponential of x.
func Exp(x float64) float64 { return fastmath.FastExp(x) }

// Log returns the natural logarithm of x.
func Log(x float64) float64 { return fastmath.FastLog(x) }

// Sin returns the sine of the radian argument x.
func Sin(x float64) float64 { return fastmath.Sin(x) }

// Cos returns the cosine of the radian argument x.
func Cos(x float64) float64 { return fastmath.Cos(x) }

// Sqrt returns the square root of x.
func Sqrt(x float64) float64 { return fastmath.Sqrt(x) }

// The remaining functions have no fastmath equivalent and use scalar emlgo.

// Log2 returns the binary logarithm of x.
func Log2(x float64) float64 { return arithmetic.LogBase2(x) }

// Log10 returns the decimal logarithm of x.
func Log10(x float64) float64 { return arithmetic.LogBase10(x) }

// Pow returns x**y, the base-x exponential of y.
func Pow(x, y float64) float64 { return arithmetic.Pow(x, y) }

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
