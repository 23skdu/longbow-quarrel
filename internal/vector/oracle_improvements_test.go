package vector

import (
	"math"
	"testing"
)

func TestL2Uint8MatchesFloatReference(t *testing.T) {
	a := &Uint8Vector{Data: []byte{0, 10, 20, 30}, Scale: 0.5}
	b := &Uint8Vector{Data: []byte{5, 5, 5, 5}, Scale: 0.25}
	got := L2Uint8(a, b)
	var want float32
	for i := range a.Data {
		da := float32(a.Data[i]) * a.Scale
		db := float32(b.Data[i]) * b.Scale
		d := da - db
		want += d * d
	}
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Fatalf("L2Uint8 = %g, want %g", got, want)
	}
}

func TestCosineFloat32UsesDispatch(t *testing.T) {
	a := make([]float32, 32)
	b := make([]float32, 32)
	for i := range a {
		a[i] = float32(i + 1)
		b[i] = float32(32 - i)
	}
	got := CosineFloat32(a, b)
	want := CosineFloat32Dispatch(a, b)
	if got != want {
		t.Fatalf("CosineFloat32 = %g, dispatch = %g", got, want)
	}
}
