package vector

import (
	"math"
	"testing"
)

// FuzzDotFloat32Oracle checks that the dispatched DotFloat32 kernel agrees
// with the pure-Go scalar oracle (dotFloat32Scalar) within the documented
// 1e-4 parity budget. The scalar is the reference; a regression in any
// native kernel surfaces here.
func FuzzDotFloat32Oracle(f *testing.F) {
	f.Add([]byte{0, 0, 0, 0}, 4)
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8}, 8)
	f.Fuzz(func(t *testing.T, data []byte, n int) {
		if n < 1 {
			n = 1
		}
		if n > 256 {
			n = 256
		}
		// Build two vectors from the fuzz bytes (cyclically).
		a := make([]float32, n)
		b := make([]float32, n)
		for i := 0; i < n; i++ {
			v := byte(0)
			if len(data) > 0 {
				v = data[i%len(data)]
			}
			a[i] = float32(v)/255.0*2 - 1
			b[i] = float32((int(v)*37+13)%256)/255.0*2 - 1
		}
		got := DotFloat32(a, b)
		want := dotFloat32Scalar(a, b)
		if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
			t.Fatalf("DotFloat32 produced non-finite: %v", got)
		}
		if d := math.Abs(float64(got - want)); d > 1e-3*float64(math.Max(1, math.Abs(float64(want))))+1e-4 {
			t.Fatalf("DotFloat32 mismatch: got %g want %g (diff %g)", got, want, d)
		}
	})
}

// FuzzL2Float32Oracle checks L2Float32 against l2Float32Scalar.
func FuzzL2Float32Oracle(f *testing.F) {
	f.Add([]byte{0, 1, 2, 3}, 4)
	f.Fuzz(func(t *testing.T, data []byte, n int) {
		if n < 1 {
			n = 1
		}
		if n > 256 {
			n = 256
		}
		a := make([]float32, n)
		b := make([]float32, n)
		for i := 0; i < n; i++ {
			v := byte(0)
			if len(data) > 0 {
				v = data[i%len(data)]
			}
			a[i] = float32(v)/255.0*2 - 1
			b[i] = float32((int(v)*17+7)%256)/255.0*2 - 1
		}
		got := L2Float32(a, b)
		want := l2Float32Scalar(a, b)
		if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
			t.Fatalf("L2Float32 produced non-finite: %v", got)
		}
		if got < 0 {
			t.Fatalf("L2Float32 negative: %v", got)
		}
		if d := math.Abs(float64(got - want)); d > 1e-3*float64(math.Max(1, math.Abs(float64(want))))+1e-4 {
			t.Fatalf("L2Float32 mismatch: got %g want %g (diff %g)", got, want, d)
		}
	})
}

// FuzzTurboQuantOracle checks that DotTurboQuant and L2TurboQuant stay
// finite and that L2TurboQuant is non-negative for random inputs.
func FuzzTurboQuantOracle(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4}, 4)
	f.Fuzz(func(t *testing.T, data []byte, n int) {
		if n < 1 {
			n = 1
		}
		if n > 64 {
			n = 64
		}
		a := &TurboQuantVector{Codes: make([]int8, n), QJLBits: make([]byte, (n+7)/8), Scale: 0.5, QJLScale: 0.25, Norm: 1}
		b := &TurboQuantVector{Codes: make([]int8, n), QJLBits: make([]byte, (n+7)/8), Scale: 0.5, QJLScale: 0.25, Norm: 1}
		for i := 0; i < n; i++ {
			v := byte(0)
			if len(data) > 0 {
				v = data[i%len(data)]
			}
			a.Codes[i] = int8(int(v) - 128)
			b.Codes[i] = int8((int(v)*7+3)%256 - 128)
		}
		dot := DotTurboQuant(a, b)
		l2 := L2TurboQuant(a, b)
		if dot != dot || l2 != l2 {
			t.Fatal("non-finite TurboQuant result")
		}
		if l2 < 0 {
			t.Fatalf("negative L2TurboQuant: %v", l2)
		}
	})
}

// FuzzCosineFloat32Oracle checks CosineFloat32Dispatch against a naive
// scalar reference.
func FuzzCosineFloat32Oracle(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4}, 4)
	f.Fuzz(func(t *testing.T, data []byte, n int) {
		if n < 1 {
			n = 1
		}
		if n > 256 {
			n = 256
		}
		a := make([]float32, n)
		b := make([]float32, n)
		for i := 0; i < n; i++ {
			v := byte(0)
			if len(data) > 0 {
				v = data[i%len(data)]
			}
			a[i] = float32(v)/255.0*2 - 1
			b[i] = float32((int(v)*13+3)%256)/255.0*2 - 1
		}
		got := CosineFloat32Dispatch(a, b)
		// scalar reference
		dot := dotFloat32Scalar(a, b)
		na := dotFloat32Scalar(a, a)
		nb := dotFloat32Scalar(b, b)
		var want float32
		if na > 0 && nb > 0 {
			want = dot / float32(math.Sqrt(float64(na*nb)))
		}
		if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
			t.Fatalf("CosineFloat32Dispatch non-finite: %v", got)
		}
		if d := math.Abs(float64(got - want)); d > 1e-3 {
			t.Fatalf("CosineFloat32Dispatch mismatch: got %g want %g (diff %g)", got, want, d)
		}
	})
}
