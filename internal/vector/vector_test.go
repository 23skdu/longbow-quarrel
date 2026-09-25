package vector

import (
	"math"
	"testing"
)

func TestDistanceFloat32(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	b := []float32{9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0}

	dot := DotFloat32(a, b)
	// 9 + 16 + 21 + 24 + 25 + 24 + 21 + 16 + 9 = 165
	expectedDot := float32(165.0)
	if math.Abs(float64(dot-expectedDot)) > 1e-4 {
		t.Errorf("DotFloat32: expected %f, got %f", expectedDot, dot)
	}

	l2 := L2Float32(a, b)
	// (8^2 + 6^2 + 4^2 + 2^2 + 0^2 + 2^2 + 4^2 + 6^2 + 8^2) = 64 + 36 + 16 + 4 + 0 + 4 + 16 + 36 + 64 = 240
	expectedL2 := float32(240.0)
	if math.Abs(float64(l2-expectedL2)) > 1e-4 {
		t.Errorf("L2Float32: expected %f, got %f", expectedL2, l2)
	}

	cos := CosineFloat32(a, a)
	if math.Abs(float64(cos-1.0)) > 1e-4 {
		t.Errorf("CosineFloat32 self: expected 1.0, got %f", cos)
	}

	// Empty slices
	if DotFloat32(nil, nil) != 0 {
		t.Errorf("Expected 0 on nil")
	}
	if CosineFloat32([]float32{0, 0}, []float32{0, 0}) != 0 {
		t.Errorf("Expected 0 on zero norms")
	}
}

func TestDistanceComplex128(t *testing.T) {
	a := []complex128{complex(1, 2), complex(3, 4), complex(5, 6), complex(7, 8), complex(9, 10)}
	b := []complex128{complex(2, 1), complex(4, 3), complex(6, 5), complex(8, 7), complex(10, 9)}

	dot := DotComplex128(a, b)
	if dot <= 0 {
		t.Errorf("Expected positive Hermitian dot, got %f", dot)
	}

	l2 := L2Complex128(a, b)
	if l2 <= 0 {
		t.Errorf("Expected positive L2 distance, got %f", l2)
	}

	cos := CosineComplex128(a, a)
	if math.Abs(cos-1.0) > 1e-4 {
		t.Errorf("Expected CosineComplex128 self = 1.0, got %f", cos)
	}

	if CosineComplex128(nil, nil) != 0 {
		t.Errorf("Expected 0 on nil")
	}
}

func TestDistanceUint8(t *testing.T) {
	a := &Uint8Vector{
		ID:    0,
		Data:  []uint8{10, 20, 30, 40, 50},
		Scale: 0.1,
	}
	b := &Uint8Vector{
		ID:    1,
		Data:  []uint8{50, 40, 30, 20, 10},
		Scale: 0.1,
	}

	dot := DotUint8(a, b)
	if dot <= 0 {
		t.Errorf("Expected positive dot product, got %f", dot)
	}

	l2 := L2Uint8(a, b)
	if l2 <= 0 {
		t.Errorf("Expected positive L2 distance, got %f", l2)
	}

	cos := CosineUint8(a, a)
	if math.Abs(float64(cos-1.0)) > 1e-4 {
		t.Errorf("Expected CosineUint8 self = 1.0, got %f", cos)
	}

	zeroVec := &Uint8Vector{Data: []uint8{0, 0}, Scale: 1}
	if CosineUint8(zeroVec, zeroVec) != 0 {
		t.Errorf("Expected 0 for zero vector")
	}
}

func TestDistanceTurboQuant(t *testing.T) {
	a := &TurboQuantVector{
		ID:       0,
		Codes:    []int8{1, -2, 3, -4, 5, -6, 7, -8, 2},
		Scale:    0.2,
		QJLBits:  []byte{0xAA, 0x55},
		QJLScale: 0.05,
		Norm:     1.0,
	}
	b := &TurboQuantVector{
		ID:       1,
		Codes:    []int8{2, -1, 4, -3, 6, -5, 8, -7, 1},
		Scale:    0.2,
		QJLBits:  []byte{0xAA, 0x55},
		QJLScale: 0.05,
		Norm:     1.0,
	}

	dot := DotTurboQuant(a, b)
	if dot <= 0 {
		t.Errorf("Expected positive dot product, got %f", dot)
	}

	l2 := L2TurboQuant(a, b)
	if l2 < 0 {
		t.Errorf("Expected non-negative L2 distance, got %f", l2)
	}

	cos := CosineTurboQuant(a, a)
	if cos <= 0 {
		t.Errorf("Expected positive cosine similarity, got %f", cos)
	}

	zeroTQ := &TurboQuantVector{Norm: 0}
	if CosineTurboQuant(zeroTQ, zeroTQ) != 0 {
		t.Errorf("Expected 0 for zero norm")
	}
}

func TestQuantizers(t *testing.T) {
	vec := []float32{0.1, -0.5, 0.8, -0.2, 0.4}
	u8 := QuantizeUint8(vec, 42)
	if u8.ID != 42 || len(u8.Data) != 5 {
		t.Errorf("QuantizeUint8 failed: %+v", u8)
	}

	emptyU8 := QuantizeUint8([]float32{}, 1)
	if len(emptyU8.Data) != 0 {
		t.Errorf("Expected empty uint8 data")
	}

	comp := Float32ToComplex128(vec)
	if len(comp) != 5 || real(comp[0]) != float64(vec[0]) {
		t.Errorf("Float32ToComplex128 failed: %+v", comp)
	}

	rot := CreateIdentityMatrix(5)
	qjl := CreateRandomQJLMatrix(8, 5, 123)
	tq := QuantizeTurboQuant(vec, 99, rot, qjl, 8)
	if tq.ID != 99 || len(tq.Codes) != 5 || len(tq.QJLBits) != 1 {
		t.Errorf("QuantizeTurboQuant failed: %+v", tq)
	}

	emptyTQ := QuantizeTurboQuant([]float32{}, 0, nil, nil, 0)
	if emptyTQ.ID != 0 {
		t.Errorf("Expected empty TQ")
	}

	dataset := GenerateRandomFloat32Dataset(10, 16, 42)
	if len(dataset) != 10 || len(dataset[0]) != 16 {
		t.Fatalf("GenerateRandomFloat32Dataset failed")
	}
}

func TestFlatIndex(t *testing.T) {
	dim := 8
	dataset := GenerateRandomFloat32Dataset(20, dim, 42)

	// 1. Float32 search
	fIdx := NewFlatIndex(dim, TypeFloat32)
	fIdx.AddFloat32(dataset)

	if fIdx.Count() != 20 {
		t.Errorf("Expected count=20, got %d", fIdx.Count())
	}

	resDot, err := fIdx.Search(dataset[0], 5, MetricDot)
	if err != nil || len(resDot) != 5 {
		t.Fatalf("Float32 MetricDot search failed: %v", err)
	}
	if resDot[0].ID != 0 {
		t.Errorf("Expected nearest neighbor to be itself (ID=0), got %d", resDot[0].ID)
	}

	resCosine, err := fIdx.Search(dataset[0], 5, MetricCosine)
	if err != nil || len(resCosine) != 5 {
		t.Fatalf("Float32 MetricCosine search failed: %v", err)
	}

	resL2, err := fIdx.Search(dataset[0], 5, MetricL2)
	if err != nil || len(resL2) != 5 {
		t.Fatalf("Float32 MetricL2 search failed: %v", err)
	}

	// 2. Uint8 search
	uIdx := NewFlatIndex(dim, TypeUint8)
	uVecs := make([]Uint8Vector, len(dataset))
	for i, v := range dataset {
		uVecs[i] = QuantizeUint8(v, i)
	}
	uIdx.AddUint8(uVecs)
	if uIdx.Count() != 20 {
		t.Errorf("Expected count=20, got %d", uIdx.Count())
	}
	resU8, err := uIdx.Search(&uVecs[0], 5, MetricDot)
	if err != nil || len(resU8) != 5 {
		t.Fatalf("Uint8 Search failed: %v", err)
	}
	found := false
	for _, r := range resU8 {
		if r.ID == 0 {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("Uint8: expected ID=0 in top-5 results, got %+v", resU8)
	}

	// 3. Complex128 search
	cIdx := NewFlatIndex(dim, TypeComplex128)
	cVecs := make([][]complex128, len(dataset))
	for i, v := range dataset {
		cVecs[i] = Float32ToComplex128(v)
	}
	cIdx.AddComplex128(cVecs)
	if cIdx.Count() != 20 {
		t.Errorf("Expected count=20, got %d", cIdx.Count())
	}
	resC, err := cIdx.Search(cVecs[0], 5, MetricDot)
	if err != nil || len(resC) != 5 {
		t.Fatalf("Complex128 Search failed: %v", err)
	}
	if resC[0].ID != 0 {
		t.Errorf("Complex128: expected nearest neighbor ID=0, got %d", resC[0].ID)
	}

	// 4. TurboQuant search
	tqIdx := NewFlatIndex(dim, TypeTurboQuant)
	rot := CreateIdentityMatrix(dim)
	qjl := CreateRandomQJLMatrix(16, dim, 99)
	tqVecs := make([]TurboQuantVector, len(dataset))
	for i, v := range dataset {
		tqVecs[i] = QuantizeTurboQuant(v, i, rot, qjl, 16)
	}
	tqIdx.AddTurboQuant(tqVecs)
	if tqIdx.Count() != 20 {
		t.Errorf("Expected count=20, got %d", tqIdx.Count())
	}
	resTQ, err := tqIdx.Search(&tqVecs[0], 5, MetricDot)
	if err != nil || len(resTQ) != 5 {
		t.Fatalf("TurboQuant Search failed: %v", err)
	}
	if resTQ[0].ID != 0 {
		t.Errorf("TurboQuant: expected nearest neighbor ID=0, got %d", resTQ[0].ID)
	}

	// Empty and edge cases
	emptyIdx := NewFlatIndex(dim, TypeFloat32)
	resEmpty, _ := emptyIdx.Search(dataset[0], 5, MetricDot)
	if resEmpty != nil {
		t.Errorf("Expected nil on empty index")
	}

	resZeroK, _ := fIdx.Search(dataset[0], 0, MetricDot)
	if resZeroK != nil {
		t.Errorf("Expected nil on k=0")
	}
}

func TestIVFIndex(t *testing.T) {
	dim := 8
	dataset := GenerateRandomFloat32Dataset(50, dim, 42)

	ivf := NewIVFIndex(dim, 4, 2, TypeFloat32)
	if err := ivf.Build(dataset); err != nil {
		t.Fatalf("IVF Build failed: %v", err)
	}

	res, err := ivf.Search(dataset[0], 5)
	if err != nil || len(res) == 0 {
		t.Fatalf("IVF Search failed: %v", err)
	}

	// Test unbuilt search
	unbuilt := NewIVFIndex(dim, 4, 2, TypeFloat32)
	if _, err := unbuilt.Search(dataset[0], 5); err == nil {
		t.Errorf("Expected error searching unbuilt IVF index")
	}

	// Test Uint8, Complex, TurboQuant branches
	ivfU8 := NewIVFIndex(dim, 4, 2, TypeUint8)
	uVecs := make([]Uint8Vector, len(dataset))
	for i, v := range dataset {
		uVecs[i] = QuantizeUint8(v, i)
	}
	ivfU8.Uint8 = uVecs
	_ = ivfU8.Build(dataset)
	resU8, err := ivfU8.Search(dataset[0], 5)
	if err != nil || len(resU8) == 0 {
		t.Errorf("IVF Uint8 Search failed: %v", err)
	}

	ivfC := NewIVFIndex(dim, 4, 2, TypeComplex128)
	cVecs := make([][]complex128, len(dataset))
	for i, v := range dataset {
		cVecs[i] = Float32ToComplex128(v)
	}
	ivfC.Complex = cVecs
	_ = ivfC.Build(dataset)
	resC, err := ivfC.Search(dataset[0], 5)
	if err != nil || len(resC) == 0 {
		t.Errorf("IVF Complex Search failed: %v", err)
	}

	ivfTQ := NewIVFIndex(dim, 4, 2, TypeTurboQuant)
	rot := CreateIdentityMatrix(dim)
	qjl := CreateRandomQJLMatrix(16, dim, 99)
	tqVecs := make([]TurboQuantVector, len(dataset))
	for i, v := range dataset {
		tqVecs[i] = QuantizeTurboQuant(v, i, rot, qjl, 16)
	}
	ivfTQ.Turbo = tqVecs
	_ = ivfTQ.Build(dataset)
	resTQ, err := ivfTQ.Search(dataset[0], 5)
	if err != nil || len(resTQ) == 0 {
		t.Errorf("IVF TurboQuant Search failed: %v", err)
	}
}

func TestHNSWIndex(t *testing.T) {
	dim := 8
	dataset := GenerateRandomFloat32Dataset(40, dim, 42)

	hnsw := NewHNSWIndex(dim, 8, 16, TypeFloat32)
	if err := hnsw.Build(dataset); err != nil {
		t.Fatalf("HNSW Build failed: %v", err)
	}

	res, err := hnsw.Search(dataset[0], 5)
	if err != nil || len(res) == 0 {
		t.Fatalf("HNSW Search failed: %v", err)
	}

	// Test unbuilt search
	unbuilt := NewHNSWIndex(dim, 8, 16, TypeFloat32)
	if _, err := unbuilt.Search(dataset[0], 5); err == nil {
		t.Errorf("Expected error searching empty HNSW")
	}

	// Test Uint8, Complex, TurboQuant branches
	hnswU8 := NewHNSWIndex(dim, 8, 16, TypeUint8)
	uVecs := make([]Uint8Vector, len(dataset))
	for i, v := range dataset {
		uVecs[i] = QuantizeUint8(v, i)
	}
	hnswU8.Uint8 = uVecs
	_ = hnswU8.Build(dataset)
	resU8, err := hnswU8.Search(dataset[0], 5)
	if err != nil || len(resU8) == 0 {
		t.Errorf("HNSW Uint8 Search failed: %v", err)
	}

	hnswC := NewHNSWIndex(dim, 8, 16, TypeComplex128)
	cVecs := make([][]complex128, len(dataset))
	for i, v := range dataset {
		cVecs[i] = Float32ToComplex128(v)
	}
	hnswC.Complex = cVecs
	_ = hnswC.Build(dataset)
	resC, err := hnswC.Search(dataset[0], 5)
	if err != nil || len(resC) == 0 {
		t.Errorf("HNSW Complex Search failed: %v", err)
	}

	hnswTQ := NewHNSWIndex(dim, 8, 16, TypeTurboQuant)
	rot := CreateIdentityMatrix(dim)
	qjl := CreateRandomQJLMatrix(16, dim, 99)
	tqVecs := make([]TurboQuantVector, len(dataset))
	for i, v := range dataset {
		tqVecs[i] = QuantizeTurboQuant(v, i, rot, qjl, 16)
	}
	hnswTQ.Turbo = tqVecs
	_ = hnswTQ.Build(dataset)
	resTQ, err := hnswTQ.Search(dataset[0], 5)
	if err != nil || len(resTQ) == 0 {
		t.Errorf("HNSW TurboQuant Search failed: %v", err)
	}
}
