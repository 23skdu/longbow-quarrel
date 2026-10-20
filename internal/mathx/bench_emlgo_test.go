//go:build emlgo

package mathx

import (
	"testing"

	"github.com/emlgo/eml/pkg/arithmetic"
	"github.com/emlgo/eml/pkg/hyper"
	"github.com/emlgo/eml/pkg/logexp"
	"github.com/emlgo/eml/pkg/trig"
)

// These benchmarks compare the scalar emlgo path (a Go loop calling the scalar
// function per element) against emlgo's SIMD batch path (one call processing
// the whole slice). The scalar path is what the engine uses today; the batch
// path is the potential win emlgo exists to deliver. They are only compiled
// under -tags emlgo.

const benchLen = 1024

func benchSlice() []float64 {
	x := make([]float64, benchLen)
	for i := range x {
		x[i] = float64(i%1000) + 1
	}
	return x
}

func scalarExp(b *testing.B) {
	x := benchSlice()
	for i := 0; i < b.N; i++ {
		for _, v := range x {
			_ = Exp(v)
		}
	}
}

func batchExp(b *testing.B) {
	x := benchSlice()
	for i := 0; i < b.N; i++ {
		_ = logexp.ExpBatch(x)
	}
}

func BenchmarkScalarExp(b *testing.B) { scalarExp(b) }
func BenchmarkBatchExp(b *testing.B)  { batchExp(b) }

func scalarLog(b *testing.B) {
	x := benchSlice()
	for i := 0; i < b.N; i++ {
		for _, v := range x {
			_ = Log(v)
		}
	}
}

func batchLog(b *testing.B) {
	x := benchSlice()
	for i := 0; i < b.N; i++ {
		_ = logexp.LogBatch(x)
	}
}

func BenchmarkScalarLog(b *testing.B) { scalarLog(b) }
func BenchmarkBatchLog(b *testing.B)  { batchLog(b) }

func scalarSin(b *testing.B) {
	x := benchSlice()
	for i := 0; i < b.N; i++ {
		for _, v := range x {
			_ = Sin(v)
		}
	}
}

func batchSin(b *testing.B) {
	x := benchSlice()
	for i := 0; i < b.N; i++ {
		_ = trig.SinBatch(x)
	}
}

func BenchmarkScalarSin(b *testing.B) { scalarSin(b) }
func BenchmarkBatchSin(b *testing.B)  { batchSin(b) }

func scalarCos(b *testing.B) {
	x := benchSlice()
	for i := 0; i < b.N; i++ {
		for _, v := range x {
			_ = Cos(v)
		}
	}
}

func batchCos(b *testing.B) {
	x := benchSlice()
	for i := 0; i < b.N; i++ {
		_ = trig.CosBatch(x)
	}
}

func BenchmarkScalarCos(b *testing.B) { scalarCos(b) }
func BenchmarkBatchCos(b *testing.B)  { batchCos(b) }

func scalarTanh(b *testing.B) {
	x := benchSlice()
	for i := 0; i < b.N; i++ {
		for _, v := range x {
			_ = Tanh(v)
		}
	}
}

func batchTanh(b *testing.B) {
	x := benchSlice()
	for i := 0; i < b.N; i++ {
		_ = hyper.TanhBatch(x)
	}
}

func BenchmarkScalarTanh(b *testing.B) { scalarTanh(b) }
func BenchmarkBatchTanh(b *testing.B)  { batchTanh(b) }

func scalarSqrt(b *testing.B) {
	x := benchSlice()
	for i := 0; i < b.N; i++ {
		for _, v := range x {
			_ = Sqrt(v)
		}
	}
}

func batchSqrt(b *testing.B) {
	x := benchSlice()
	for i := 0; i < b.N; i++ {
		_ = arithmetic.SqrtBatch(x)
	}
}

func BenchmarkScalarSqrt(b *testing.B) { scalarSqrt(b) }
func BenchmarkBatchSqrt(b *testing.B)  { batchSqrt(b) }
