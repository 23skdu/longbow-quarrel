//go:build !emlgo

package mathx

import "math"

// This file is the default backend. Every function delegates directly to the
// Go standard library math package, so the !emlgo build is bit-identical to a
// tree that imports "math" directly.

// Exp returns e**x, the base-e exponential of x.
func Exp(x float64) float64 { return math.Exp(x) }

// Log returns the natural logarithm of x.
func Log(x float64) float64 { return math.Log(x) }

// Log2 returns the binary logarithm of x.
func Log2(x float64) float64 { return math.Log2(x) }

// Log10 returns the decimal logarithm of x.
func Log10(x float64) float64 { return math.Log10(x) }

// Sqrt returns the square root of x.
func Sqrt(x float64) float64 { return math.Sqrt(x) }

// Pow returns x**y, the base-x exponential of y.
func Pow(x, y float64) float64 { return math.Pow(x, y) }

// Sin returns the sine of the radian argument x.
func Sin(x float64) float64 { return math.Sin(x) }

// Cos returns the cosine of the radian argument x.
func Cos(x float64) float64 { return math.Cos(x) }

// Tanh returns the hyperbolic tangent of x.
func Tanh(x float64) float64 { return math.Tanh(x) }

// Abs returns the absolute value of x.
func Abs(x float64) float64 { return math.Abs(x) }

// Floor returns the greatest integer value less than or equal to x.
func Floor(x float64) float64 { return math.Floor(x) }

// Ceil returns the least integer value greater than or equal to x.
func Ceil(x float64) float64 { return math.Ceil(x) }

// Round returns the nearest integer, rounding half away from zero.
func Round(x float64) float64 { return math.Round(x) }

// Trunc returns the integer value of x.
func Trunc(x float64) float64 { return math.Trunc(x) }
