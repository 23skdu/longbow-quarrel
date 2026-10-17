// Package mathx provides a pluggable math backend for the engine.
//
// The compute functions (Exp, Log, Sqrt, Pow, Sin, Cos, ...) are dispatched to
// either the Go standard library math package (the default) or emlgo, selected
// at build time with the "emlgo" build tag. The control-flow, bit-manipulation,
// and constant helpers (IsNaN, IsInf, NaN, Inf, Float32bits, ...) have no emlgo
// equivalent and always delegate to the standard library.
//
// This is the only package in the tree that imports "math" or emlgo; every
// other package calls through mathx so the backend can be A/B tested with:
//
//	go test -tags emlgo ./...
//
// An additional "emlfast" tag swaps in emlgo's minimax-polynomial fastmath
// approximations (not bit-exact) to measure the speed/accuracy trade-off.
package mathx

import "math"

// IsNaN reports whether f is an IEEE 754 "not-a-number" value.
func IsNaN(f float64) bool { return math.IsNaN(f) }

// IsInf reports whether f is an infinity, according to sign.
func IsInf(f float64, sign int) bool { return math.IsInf(f, sign) }

// NaN returns an IEEE 754 "not-a-number" value.
func NaN() float64 { return math.NaN() }

// Inf returns positive infinity if sign >= 0, negative infinity if sign < 0.
func Inf(sign int) float64 { return math.Inf(sign) }

// Float32bits returns the IEEE 754 binary representation of f.
func Float32bits(f float32) uint32 { return math.Float32bits(f) }

// Float32frombits returns the floating-point number corresponding to the
// IEEE 754 binary representation b.
func Float32frombits(b uint32) float32 { return math.Float32frombits(b) }

// Float64bits returns the IEEE 754 binary representation of f.
func Float64bits(f float64) uint64 { return math.Float64bits(f) }

// Float64frombits returns the floating-point number corresponding to the
// IEEE 754 binary representation b.
func Float64frombits(b uint64) float64 { return math.Float64frombits(b) }

// Copysign returns a value with the magnitude of f and the sign of sign.
func Copysign(f, sign float64) float64 { return math.Copysign(f, sign) }

// Max returns the larger of x or y.
//
// This delegates to math.Max rather than emlgo's arithmetic.Max because the
// two differ in NaN handling: math.Max propagates NaN according to the IEEE
// 754 / Go spec (e.g. Max(NaN, +Inf) = +Inf, Max(NaN, -Inf) = NaN), while
// emlgo returns the non-NaN operand. The engine relies on stdlib NaN
// propagation for its NaN guards, so Max/Min always use the standard library.
func Max(x, y float64) float64 { return math.Max(x, y) }

// Min returns the smaller of x or y. See Max for why this uses math.Min.
func Min(x, y float64) float64 { return math.Min(x, y) }

// MaxInt is the maximum value of the int type.
const MaxInt = math.MaxInt

// MaxFloat32 is the maximum finite value of the float32 type.
const MaxFloat32 = math.MaxFloat32

// MaxFloat64 is the maximum finite value of the float64 type.
const MaxFloat64 = math.MaxFloat64

// SmallestNonzeroFloat32 is the smallest positive, non-zero value of the
// float32 type.
const SmallestNonzeroFloat32 = math.SmallestNonzeroFloat32

// Pi is the mathematical constant π.
const Pi = math.Pi
