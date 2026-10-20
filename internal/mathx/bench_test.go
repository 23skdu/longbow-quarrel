package mathx

import "testing"

// These benchmarks are the A/B harness: run them with and without -tags emlgo
// (and -tags "emlgo emlfast") to compare the stdlib and emlgo backends.
//
//	go test -run '^$' -bench . -benchmem ./internal/mathx/
//	go test -tags emlgo -run '^$' -bench . -benchmem ./internal/mathx/

func BenchmarkExp(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Exp(float64(i%1000) * 0.001)
	}
}

func BenchmarkLog(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Log(float64(i%1000) + 1)
	}
}

func BenchmarkLog2(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Log2(float64(i%1000) + 1)
	}
}

func BenchmarkLog10(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Log10(float64(i%1000) + 1)
	}
}

func BenchmarkSqrt(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Sqrt(float64(i%1000) + 1)
	}
}

func BenchmarkPow(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Pow(float64(i%100)+1, 0.5)
	}
}

func BenchmarkSin(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Sin(float64(i%1000) * 0.001)
	}
}

func BenchmarkCos(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Cos(float64(i%1000) * 0.001)
	}
}

func BenchmarkTanh(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Tanh(float64(i%1000) * 0.001)
	}
}

func BenchmarkAbs(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Abs(float64(i%1000) - 500)
	}
}

func BenchmarkMax(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Max(float64(i%1000), 500)
	}
}

func BenchmarkMin(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Min(float64(i%1000), 500)
	}
}

func BenchmarkFloor(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Floor(float64(i%1000) * 0.001)
	}
}

func BenchmarkCeil(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Ceil(float64(i%1000) * 0.001)
	}
}

func BenchmarkRound(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Round(float64(i%1000) * 0.001)
	}
}

func BenchmarkTrunc(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Trunc(float64(i%1000) * 0.001)
	}
}
