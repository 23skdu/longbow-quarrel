package gguf

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/mathx"
)

func TestDequantFloat16AndBF16Coverage(t *testing.T) {
	// Float16ToFloat32 branches
	// 1. Zero: exp == 0, frac == 0
	if Float16ToFloat32(0x0000) != 0.0 {
		t.Errorf("expected 0.0")
	}
	// 2. Negative Zero: exp == 0, frac == 0, sign != 0
	if mathx.Float32bits(Float16ToFloat32(0x8000)) != 0x80000000 {
		t.Errorf("expected -0.0")
	}
	// 3. Subnormal positive: exp == 0, frac != 0
	if Float16ToFloat32(0x0001) <= 0 {
		t.Errorf("expected > 0")
	}
	// 4. Subnormal negative: exp == 0, frac != 0, sign != 0
	if Float16ToFloat32(0x8001) >= 0 {
		t.Errorf("expected < 0")
	}
	// 5. Positive Inf: exp == 0x1F, frac == 0, sign == 0
	pInf := Float16ToFloat32(0x7C00)
	if !mathx.IsInf(float64(pInf), 1) {
		t.Errorf("expected +Inf")
	}
	// 6. Negative Inf: exp == 0x1F, frac == 0, sign != 0
	nInf := Float16ToFloat32(0xFC00)
	if !mathx.IsInf(float64(nInf), -1) {
		t.Errorf("expected -Inf")
	}
	// 7. NaN: exp == 0x1F, frac != 0
	nan := Float16ToFloat32(0x7C01)
	if !mathx.IsNaN(float64(nan)) {
		t.Errorf("expected NaN")
	}
	// 8. Normal positive and negative
	if Float16ToFloat32(0x3C00) != 1.0 { // 1.0 in f16
		t.Errorf("expected 1.0")
	}
	if Float16ToFloat32(0xBC00) != -1.0 { // -1.0 in f16
		t.Errorf("expected -1.0")
	}

	// BF16ToFloat32
	bf16Val := uint16(0x3F80) // 1.0 in BF16
	if BF16ToFloat32(bf16Val) != 1.0 {
		t.Errorf("expected 1.0 from BF16")
	}

	// DequantizeBF16
	bfData := []byte{0x80, 0x3F, 0x00, 0x40} // 1.0, 2.0
	res := DequantizeBF16(bfData, 2)
	if len(res) != 2 || res[0] != 1.0 || res[1] != 2.0 {
		t.Errorf("unexpected DequantizeBF16 result: %v", res)
	}
	// Truncated data
	resTrunc := DequantizeBF16(bfData, 4)
	if len(resTrunc) != 4 || resTrunc[2] != 0.0 {
		t.Errorf("unexpected DequantizeBF16 truncated result: %v", resTrunc)
	}

	// MatVecMulBF16
	// cols == 0 or len(vector) < cols
	if len(MatVecMulBF16(bfData, []float32{1.0}, 2, 0)) != 2 {
		t.Errorf("expected len 2")
	}
	if len(MatVecMulBF16(bfData, []float32{}, 2, 2)) != 2 {
		t.Errorf("expected len 2")
	}
	// data too short
	if len(MatVecMulBF16([]byte{0}, []float32{1.0, 2.0}, 2, 2)) != 2 {
		t.Errorf("expected len 2")
	}
	// small rows < 32
	vec := []float32{1.0, 2.0}
	matrixData := make([]byte, 4*2*2) // 4 rows, 2 cols = 8 elements * 2 bytes = 16 bytes
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint16(matrixData[i*2:i*2+2], 0x3F80) // 1.0
	}
	outBF16Small := MatVecMulBF16(matrixData, vec, 4, 2)
	if len(outBF16Small) != 4 || outBF16Small[0] != 3.0 {
		t.Errorf("expected 3.0, got %v", outBF16Small)
	}
	// large rows >= 32 (parallel multi-thread with boundary remainder)
	largeRows := 37
	largeMatrixData := make([]byte, largeRows*2*2)
	for i := 0; i < largeRows*2; i++ {
		binary.LittleEndian.PutUint16(largeMatrixData[i*2:i*2+2], 0x3F80)
	}
	outBF16Large := MatVecMulBF16(largeMatrixData, vec, largeRows, 2)
	if len(outBF16Large) != largeRows || outBF16Large[0] != 3.0 {
		t.Errorf("expected 3.0 for large BF16, got %v", outBF16Large[0])
	}
}

func TestDequantPanicsAndTruncations(t *testing.T) {
	assertPanic := func(name string, f func()) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("%s: expected panic, got none", name)
			}
		}()
		f()
	}

	assertPanic("DequantizeQ5K mod 256", func() { DequantizeQ5K(nil, 100) })
	assertPanic("DequantizeQ4K mod 256", func() { DequantizeQ4K(nil, 100) })
	assertPanic("DequantizeQ4KBranchless mod 256", func() { DequantizeQ4KBranchless(nil, 100) })
	assertPanic("DequantizeQ8_0 mod 32", func() { DequantizeQ8_0(nil, 15) })
	assertPanic("DequantizeQ4_0 mod 32", func() { DequantizeQ4_0(nil, 15) })
	assertPanic("DequantizeQ5_0 mod 32", func() { DequantizeQ5_0(nil, 15) })
	assertPanic("DequantizeQ4K_SIMD mod 256", func() { DequantizeQ4K_SIMD(nil, 100) })
	assertPanic("DequantizeQ6K_SIMD mod 256", func() { DequantizeQ6K_SIMD(nil, 100) })

	// Truncated data checks (break paths for functions that check block bounds)
	shortBuf := make([]byte, 10)
	_ = DequantizeQ5K(shortBuf, 256)
	_ = DequantizeQ3K(shortBuf, 256)
	_ = DequantizeQ2K(shortBuf, 256)
	_ = DequantizeIQ4XS(shortBuf, 256)
	_ = DequantizeQ6K(shortBuf, 256)
	_ = DequantizeQ8_0(shortBuf, 32)
	_ = DequantizeQ4_0(shortBuf, 32)
	_ = DequantizeQ5_0(shortBuf, 32)
	_ = DequantizeIQ4NL(shortBuf, 32)
	_ = DequantizeQ80(shortBuf, 32)
	// IQ4NL and Q80 with invalid numElements
	if len(DequantizeIQ4NL(shortBuf, 15)) != 15 {
		t.Errorf("expected 15")
	}
	if len(DequantizeQ80(shortBuf, 15)) != 15 {
		t.Errorf("expected 15")
	}

	// DequantizeF16
	f16Data := []byte{0x00, 0x3C, 0x00, 0x40} // 1.0, 2.0 in f16
	f16Out := DequantizeF16(f16Data, 2)
	if len(f16Out) != 2 || f16Out[0] != 1.0 || f16Out[1] != 2.0 {
		t.Errorf("DequantizeF16 error: %v", f16Out)
	}

}

func TestMatVecMulQ8_0Coverage(t *testing.T) {
	// cols % 32 != 0
	if len(MatVecMulQ8_0(nil, []float32{1.0}, 2, 10)) != 2 {
		t.Errorf("expected len 2")
	}
	// len(vector) < cols
	if len(MatVecMulQ8_0(nil, []float32{1.0}, 2, 32)) != 2 {
		t.Errorf("expected len 2")
	}
	// Small rows < 32
	cols := 32
	rows := 4
	data := make([]byte, rows*34)
	for r := 0; r < rows; r++ {
		binary.LittleEndian.PutUint16(data[r*34:r*34+2], 0x3C00) // 1.0
		for j := 0; j < 32; j++ {
			data[r*34+2+j] = 1
		}
	}
	vec := make([]float32, cols)
	for i := range vec {
		vec[i] = 1.0
	}
	resSmall := MatVecMulQ8_0(data, vec, rows, cols)
	if len(resSmall) != rows || resSmall[0] != 32.0 {
		t.Errorf("expected 32.0, got %v", resSmall)
	}

	// Large rows >= 32 (with boundary remainder)
	largeRows := 37
	largeData := make([]byte, largeRows*34)
	for r := 0; r < largeRows; r++ {
		binary.LittleEndian.PutUint16(largeData[r*34:r*34+2], 0x3C00)
		for j := 0; j < 32; j++ {
			largeData[r*34+2+j] = 1
		}
	}
	resLarge := MatVecMulQ8_0(largeData, vec, largeRows, cols)
	if len(resLarge) != largeRows || resLarge[0] != 32.0 {
		t.Errorf("expected 32.0, got %v", resLarge)
	}

	// Truncated data break
	shortData := make([]byte, 20)
	resTrunc := MatVecMulQ8_0(shortData, vec, rows, cols)
	if len(resTrunc) != rows || resTrunc[0] != 0 {
		t.Errorf("expected 0 on truncated data")
	}
}

func TestDequantBlockCoverage(t *testing.T) {
	dst := make([]float32, 256)
	f32Data := make([]byte, 256*4)
	for i := 0; i < 256; i++ {
		binary.LittleEndian.PutUint32(f32Data[i*4:(i+1)*4], mathx.Float32bits(float32(i)))
	}
	DequantizeBlock(f32Data, dst, GGMLTypeF32)
	if dst[1] != 1.0 {
		t.Errorf("expected 1.0")
	}
	// Truncated F32
	DequantizeBlock([]byte{1, 2}, dst, GGMLTypeF32)

	// F16
	f16Data := make([]byte, 256*2)
	for i := 0; i < 256; i++ {
		binary.LittleEndian.PutUint16(f16Data[i*2:(i+1)*2], 0x3C00)
	}
	DequantizeBlock(f16Data, dst, GGMLTypeF16)
	if dst[0] != 1.0 {
		t.Errorf("expected 1.0")
	}
	// Truncated F16
	DequantizeBlock([]byte{1}, dst, GGMLTypeF16)

	// BF16
	bfData := make([]byte, 256*2)
	for i := 0; i < 256; i++ {
		binary.LittleEndian.PutUint16(bfData[i*2:(i+1)*2], 0x3F80)
	}
	DequantizeBlock(bfData, dst, GGMLTypeBF16)
	if dst[0] != 1.0 {
		t.Errorf("expected 1.0")
	}
	// Truncated BF16
	DequantizeBlock([]byte{1}, dst, GGMLTypeBF16)

	// Other types in DequantizeBlock
	types := []GGMLType{
		GGMLTypeQ4_K, GGMLTypeQ6_K, GGMLTypeQ8_0, GGMLTypeQ5_0,
		GGMLTypeQ2_K, GGMLTypeQ3_K, GGMLTypeQ5_K, GGMLTypeIQ4_XS,
		GGMLTypeIQ4_NL, GGMLType(9999),
	}
	dummyBuf := make([]byte, 1024)
	for _, typ := range types {
		DequantizeBlock(dummyBuf, dst, typ)
	}
}

func TestDequantSIMDCoverage(t *testing.T) {
	origProcs := runtime.GOMAXPROCS(32)
	defer runtime.GOMAXPROCS(origProcs)

	numBlocks := 17
	numElements := numBlocks * 256
	q4kData := make([]byte, numBlocks*144)
	for b := 0; b < numBlocks; b++ {
		binary.LittleEndian.PutUint16(q4kData[b*144:b*144+2], 0x3C00) // d = 1.0
	}
	q6kData := make([]byte, numBlocks*210)
	for b := 0; b < numBlocks; b++ {
		binary.LittleEndian.PutUint16(q6kData[b*210+208:b*210+210], 0x3C00) // d = 1.0
	}

	// Test workers > numBlocks branch
	runtime.GOMAXPROCS(32)
	_ = DequantizeQ4K_SIMD(q4kData, numElements)
	_ = DequantizeQ6K_SIMD(q6kData, numElements)

	// Test workers < numBlocks with remainder (hits startBlock >= numBlocks and endBlock > numBlocks)
	runtime.GOMAXPROCS(12)
	outQ4K := DequantizeQ4K_SIMD(q4kData, numElements)
	if len(outQ4K) != numElements {
		t.Errorf("expected %d elements, got %d", numElements, len(outQ4K))
	}
	outQ6K := DequantizeQ6K_SIMD(q6kData, numElements)
	if len(outQ6K) != numElements {
		t.Errorf("expected %d elements, got %d", numElements, len(outQ6K))
	}

	// dequantizeQ4KBlocks and dequantizeQ6KBlocks with truncated data
	shortSlice := make([]byte, 10)
	dequantizeQ4KBlocks(shortSlice, make([]float32, 256), 0, 1)
	dequantizeQ6KBlocks(shortSlice, make([]float32, 256), 0, 1)

	// MatVecMulQ4_K and MatVecMulQ6_K
	vec := make([]float32, 256)
	for i := range vec {
		vec[i] = 1.0
	}
	// large rows
	resQ4K := MatVecMulQ4_K(q4kData, vec, numBlocks, 256)
	if len(resQ4K) != numBlocks {
		t.Errorf("expected %d results", numBlocks)
	}
	resQ6K := MatVecMulQ6_K(q6kData, vec, numBlocks, 256)
	if len(resQ6K) != numBlocks {
		t.Errorf("expected %d results", numBlocks)
	}

	// rows == 0
	_ = MatVecMulQ4_K(q4kData, vec, 0, 256)
	_ = MatVecMulQ6_K(q6kData, vec, 0, 256)

	// Truncated data in MatVecMulQ4_K and MatVecMulQ6_K
	resQ4KTrunc := MatVecMulQ4_K(q4kData[:100], vec, numBlocks, 256)
	if len(resQ4KTrunc) != numBlocks {
		t.Errorf("expected %d results", numBlocks)
	}
	resQ6KTrunc := MatVecMulQ6_K(q6kData[:100], vec, numBlocks, 256)
	if len(resQ6KTrunc) != numBlocks {
		t.Errorf("expected %d results", numBlocks)
	}

	// matVecMulGeneric edge cases
	_ = matVecMulGeneric(q4kData, vec, 0, 256, 256, 144, func(block []byte, dst []float32) {})
	_ = matVecMulGeneric(q4kData, vec, 17, 256, 256, 144, func(block []byte, dst []float32) {})

	// cols == 0, blockSize == 0, cols % blockSize != 0
	if len(matVecMulGeneric(nil, vec, 2, 0, 32, 18, nil)) != 2 {
		t.Errorf("expected len 2")
	}
	if len(matVecMulGeneric(nil, vec, 2, 32, 0, 18, nil)) != 2 {
		t.Errorf("expected len 2")
	}
	if len(matVecMulGeneric(nil, vec, 2, 30, 32, 18, nil)) != 2 {
		t.Errorf("expected len 2")
	}
	if len(matVecMulGeneric([]byte{1}, vec, 2, 32, 32, 18, nil)) != 2 {
		t.Errorf("expected len 2")
	}

	// MatVecMulQ4_0, MatVecMulIQ4_NL, MatVecMulQ5_0, MatVecMulQ2_K, MatVecMulQ3_K, MatVecMulQ5_K
	vec32 := make([]float32, 32)
	for i := range vec32 {
		vec32[i] = 1.0
	}
	dataQ4_0 := make([]byte, 2*18)
	binary.LittleEndian.PutUint16(dataQ4_0[0:2], 0x3C00)
	binary.LittleEndian.PutUint16(dataQ4_0[18:20], 0x3C00)
	_ = MatVecMulQ4_0(dataQ4_0, vec32, 2, 32)

	dataIQ4_NL := make([]byte, 2*18)
	binary.LittleEndian.PutUint16(dataIQ4_NL[0:2], 0x3C00)
	binary.LittleEndian.PutUint16(dataIQ4_NL[18:20], 0x3C00)
	_ = MatVecMulIQ4_NL(dataIQ4_NL, vec32, 2, 32)

	dataQ5_0 := make([]byte, 2*22)
	binary.LittleEndian.PutUint16(dataQ5_0[0:2], 0x3C00)
	binary.LittleEndian.PutUint16(dataQ5_0[22:24], 0x3C00)
	_ = MatVecMulQ5_0(dataQ5_0, vec32, 2, 32)

	dataQ2_K := make([]byte, 2*84)
	binary.LittleEndian.PutUint16(dataQ2_K[80:82], 0x3C00)
	binary.LittleEndian.PutUint16(dataQ2_K[84+80:84+82], 0x3C00)
	_ = MatVecMulQ2_K(dataQ2_K, vec, 2, 256)

	dataQ3_K := make([]byte, 2*110)
	binary.LittleEndian.PutUint16(dataQ3_K[108:110], 0x3C00)
	binary.LittleEndian.PutUint16(dataQ3_K[110+108:110+110], 0x3C00)
	_ = MatVecMulQ3_K(dataQ3_K, vec, 2, 256)

	dataQ5_K := make([]byte, 2*176)
	binary.LittleEndian.PutUint16(dataQ5_K[0:2], 0x3C00)
	binary.LittleEndian.PutUint16(dataQ5_K[176:178], 0x3C00)
	_ = MatVecMulQ5_K(dataQ5_K, vec, 2, 256)
}

func TestMetadataCoverage(t *testing.T) {
	// estimateMemoryUsage with 0 dimensions (size == 0) for F32, F16, and default
	f := &GGUFFile{
		DataOffset: 128,
		Tensors: []*TensorInfo{
			{Name: "t_zero_f32", Type: GGMLTypeF32, Dimensions: []uint64{0}, Offset: 128},
			{Name: "t_zero_f16", Type: GGMLTypeF16, Dimensions: []uint64{0}, Offset: 128},
			{Name: "t_zero_unk", Type: GGMLType(8888), Dimensions: []uint64{0}, Offset: 128},
		},
	}
	analyzer := NewMetadataAnalyzer(f)
	mem := analyzer.estimateMemoryUsage()
	if mem != 0 {
		t.Errorf("expected 0, got %d", mem)
	}

	// ValidateTensors with unknown size (expectedSize == 0) and offset mismatch
	issues, err := analyzer.ValidateTensors()
	if err != nil || len(issues) == 0 {
		t.Errorf("expected issues in ValidateTensors")
	}

	// FindMissingTensors
	missing := analyzer.FindMissingTensors([]string{"t_zero_f32", "missing_tensor"})
	if len(missing) != 1 || missing[0] != "missing_tensor" {
		t.Errorf("expected missing_tensor, got %v", missing)
	}

	// ComputeStats for non-existent tensor
	_, err = analyzer.ComputeStats("non_existent")
	if err == nil {
		t.Errorf("expected error for non_existent tensor")
	}

	// ComputeStats for F32 tensor with NaN, Inf, Min/Max
	f32Vals := []float32{0.0, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), -10.0, 20.0, 5.0}
	f32Bytes := make([]byte, len(f32Vals)*4)
	for i, v := range f32Vals {
		binary.LittleEndian.PutUint32(f32Bytes[i*4:], mathx.Float32bits(v))
	}
	f.Tensors = append(f.Tensors, &TensorInfo{
		Name:       "f32_stats",
		Type:       GGMLTypeF32,
		Dimensions: []uint64{uint64(len(f32Vals))},
		Data:       f32Bytes,
		Offset:     128,
	})
	stats, err := analyzer.ComputeStats("f32_stats")
	if err != nil {
		t.Fatalf("ComputeStats error: %v", err)
	}
	if !stats.HasNaN || !stats.HasInf || !mathx.IsInf(stats.MinValue, -1) || !mathx.IsInf(stats.MaxValue, 1) {
		t.Errorf("unexpected f32 stats: %+v", stats)
	}

	// ComputeStats for F16 tensor with Min/Max
	f16Vals := []uint16{0x3C00, 0x4000, 0x3800} // 1.0, 2.0, 0.5
	f16Bytes := make([]byte, len(f16Vals)*2)
	for i, v := range f16Vals {
		binary.LittleEndian.PutUint16(f16Bytes[i*2:], v)
	}
	f.Tensors = append(f.Tensors, &TensorInfo{
		Name:       "f16_stats",
		Type:       GGMLTypeF16,
		Dimensions: []uint64{uint64(len(f16Vals))},
		Data:       f16Bytes,
		Offset:     128,
	})
	f16Stats, err := analyzer.ComputeStats("f16_stats")
	if err != nil {
		t.Fatalf("ComputeStats error: %v", err)
	}
	if f16Stats.MinValue != float64(f16Vals[2]) || f16Stats.MaxValue != float64(f16Vals[1]) {
		t.Errorf("unexpected f16 stats: %+v", f16Stats)
	}
}

func TestQuantizeQ4KCoverage(t *testing.T) {
	// Float32ToFloat16
	// exp <= 0
	if Float32ToFloat16(0.0) != 0 {
		t.Errorf("expected 0")
	}
	// exp >= 31
	if Float32ToFloat16(1e38) != 0x7C00 {
		t.Errorf("expected 0x7C00 for overflow, got 0x%X", Float32ToFloat16(1e38))
	}

	// QuantizeWeightsToQ4K numElements % 256 != 0
	_, err := QuantizeWeightsToQ4K(make([]float32, 100), 100)
	if err == nil {
		t.Errorf("expected error for numElements mod 256 != 0")
	}

	// All zeros block (d == 0)
	zeroWeights := make([]float32, 256)
	qZeros, err := QuantizeWeightsToQ4K(zeroWeights, 256)
	if err != nil || len(qZeros) != 144 {
		t.Fatalf("QuantizeWeightsToQ4K failed on zeroWeights: %v", err)
	}

	// Mixed weights covering negative values, constant groups, and large values (q > 15)
	weights := make([]float32, 256)
	for i := range weights {
		if i < 32 {
			weights[i] = -5.0 // negative weights (w < groupMinVal)
		} else if i < 64 {
			weights[i] = 1.0 // constant group (rangeVal == 0)
		} else if i < 96 {
			weights[i] = 0.0001 * float32(i) // small rangeVal (sc[g] == 0 -> 1)
		} else {
			weights[i] = float32(i) * 2.0 // large values (q > 15)
		}
	}
	qWeights, err := QuantizeWeightsToQ4K(weights, 256)
	if err != nil || len(qWeights) != 144 {
		t.Fatalf("QuantizeWeightsToQ4K failed: %v", err)
	}

	// DequantizeWeightsFromQ4K
	deq, err := DequantizeWeightsFromQ4K(qWeights, 1, 256)
	if err != nil || len(deq) != 256 {
		t.Fatalf("DequantizeWeightsFromQ4K failed: %v", err)
	}
}

func TestQuantizeFP8Coverage(t *testing.T) {
	// The E4M3 and E5M2 conversions use the standard exponent biases (7
	// and 15) and the standard mantissa convention. The copy in this
	// package used to use a different convention from its own decoder, so
	// the pair did not round-trip, and it also disagreed with the copy in
	// internal/device.
	//
	// quantizeToE4M3 Inf/NaN, the saturation point, and the subnormal
	// floor.
	if quantizeToE4M3(math.Inf(1)) != 0x7F {
		t.Errorf("expected 0x7F for +Inf")
	}
	if quantizeToE4M3(math.NaN()) != 0x7F {
		t.Errorf("expected 0x7F for NaN")
	}
	// 300 is inside the E4M3 range: the exponent field is 15 and the
	// mantissa is 1, which decodes to 288.
	if quantizeToE4M3(300.0) != 0x79 {
		t.Errorf("expected 0x79 for 300, got 0x%02X", quantizeToE4M3(300.0))
	}
	if quantizeToE4M3(-300.0) != 0xF9 {
		t.Errorf("expected 0xF9 for -300, got 0x%02X", quantizeToE4M3(-300.0))
	}
	// A value far below the normal range rounds to zero.
	if got := quantizeToE4M3(1e-10); got != 0x00 {
		t.Errorf("expected zero encoding for 1e-10, got 0x%02X", got)
	}
	// The largest finite E4M3 value is 448 (exponent field 15, mantissa
	// 6), and anything above it saturates there.
	if got := quantizeToE4M3(1e9); got != 0x7E {
		t.Errorf("expected the saturation encoding 0x7E for 1e9, got 0x%02X", got)
	}
	if got := dequantizeE4M3(0x7E); math.Abs(got-448) > 0.001 {
		t.Errorf("the largest finite E4M3 value is %v, want 448", got)
	}

	// quantizeToE5M2 Inf/NaN, > 57344, < -57344, exp < -16
	if quantizeToE5M2(math.Inf(1)) != 0x7C {
		t.Errorf("expected 0x7C for +Inf")
	}
	if quantizeToE5M2(math.NaN()) != 0x7F {
		t.Errorf("expected 0x7F for NaN")
	}
	if quantizeToE5M2(60000.0) != 0x7B {
		t.Errorf("expected 0x7B for > 57344")
	}
	if quantizeToE5M2(-60000.0) != 0xFB {
		t.Errorf("expected 0xFB for < -57344")
	}
	// E5M2 clamps the exponent to its -14 subnormal floor, so 1e-20 encodes to
	// zero (sign 0, exp field 0, mantissa 0).
	if got := quantizeToE5M2(1e-20); got != 0x00 {
		t.Errorf("expected zero encoding for 1e-20, got 0x%02X", got)
	}

	// DequantizeFromFP8E4M3 zero and negative
	deqE4, _ := DequantizeFromFP8E4M3([]byte{0x00, 0x88})
	if len(deqE4) != 2 || deqE4[0] != 0 || deqE4[1] >= 0 {
		t.Errorf("expected zero and negative, got %v", deqE4)
	}

	// DequantizeFromFP8E5M2 zero and negative
	deqE5, _ := DequantizeFromFP8E5M2([]byte{0x00, 0x84})
	if len(deqE5) != 2 || deqE5[0] != 0 || deqE5[1] >= 0 {
		t.Errorf("expected zero and negative, got %v", deqE5)
	}

	// QuantizeWeightsToFP8
	// length mismatch
	_, err := QuantizeWeightsToFP8([]float32{1.0}, 2, FP8E4M3)
	if err == nil {
		t.Errorf("expected error on length mismatch")
	}
	// maxVal == 0
	qZero, err := QuantizeWeightsToFP8([]float32{0.0, 0.0}, 2, FP8E4M3)
	if err != nil || len(qZero) != 2 {
		t.Errorf("expected zero bytes, got %v", qZero)
	}
	// FP8E5M2
	qE5, err := QuantizeWeightsToFP8([]float32{1.0, -2.0}, 2, FP8E5M2)
	if err != nil || len(qE5) != 2 {
		t.Errorf("expected 2 bytes, got %v", qE5)
	}
	// Unknown FP8Type
	_, err = QuantizeWeightsToFP8([]float32{1.0, -2.0}, 2, FP8Type(99))
	if err == nil {
		t.Errorf("expected error on unknown FP8Type")
	}

	// DequantizeWeightsFromFP8
	// length mismatch
	_, err = DequantizeWeightsFromFP8([]byte{1}, 2, 2, FP8E4M3, 1.0)
	if err == nil {
		t.Errorf("expected error on length mismatch")
	}
	// FP8E5M2
	deqE5Weights, err := DequantizeWeightsFromFP8([]byte{0x20, 0x30}, 1, 2, FP8E5M2, 2.0)
	if err != nil || len(deqE5Weights) != 2 {
		t.Errorf("expected 2 elements from E5M2 dequant")
	}
	// Unknown FP8Type
	_, err = DequantizeWeightsFromFP8([]byte{0x20, 0x30}, 1, 2, FP8Type(99), 1.0)
	if err == nil {
		t.Errorf("expected error on unknown FP8Type")
	}

	// NewFP8Tensor error and maxVal == 0
	_, err = NewFP8Tensor([]float32{1.0}, []int{2, 2}, FP8E4M3)
	if err == nil {
		t.Errorf("expected error on shape mismatch")
	}
	tensorZero, err := NewFP8Tensor([]float32{0.0, 0.0}, []int{2}, FP8E4M3)
	if err != nil || tensorZero.Scale != 0 {
		t.Errorf("expected scale 0 for zero tensor")
	}
	tensorValid, err := NewFP8Tensor([]float32{1.0, 2.0}, []int{2}, FP8E4M3)
	val, errVal := tensorValid.ToFloat32()
	if err != nil || errVal != nil || len(val) != 2 || tensorValid.ByteSize() != 2 {
		t.Errorf("expected valid FP8Tensor methods")
	}

	// FP8Config UnmarshalBinary too short
	cfg := NewFP8Config(FP8E4M3)
	cfg.PerTensor = true
	cfg.ScaleFloat32 = 1.0
	cfg.BlockSize = 32
	bin, err := cfg.MarshalBinary()
	if err != nil || len(bin) != 8 {
		t.Fatalf("MarshalBinary error: %v", err)
	}
	var newCfg FP8Config
	if err := newCfg.UnmarshalBinary(bin[:4]); err == nil {
		t.Errorf("expected error on truncated binary")
	}
	if err := newCfg.UnmarshalBinary(bin); err != nil {
		t.Errorf("unexpected UnmarshalBinary error: %v", err)
	}

	// SizeBytes
	if FP8E4M3SizeBytes(10) != 10 || FP8E5M2SizeBytes(10) != 10 {
		t.Errorf("unexpected size bytes")
	}
}

func TestQuantizeTurboQuantCoverage(t *testing.T) {
	// PolarQuant size mismatch
	_, _, _, err := PolarQuant([]float32{1.0}, []float32{1.0, 2.0}, 1, 4)
	if err == nil {
		t.Errorf("expected error on PolarQuant size mismatch")
	}

	// QJLTransform size mismatch
	_, _, err = QJLTransform([]float32{1.0}, []float32{1.0}, 2, 2)
	if err == nil {
		t.Errorf("expected error on QJLTransform size mismatch")
	}

	// QuantizeTurboQuant input size not multiple of blockSize
	rot := GenerateRandomOrthogonalMatrix(32)
	qjl := GenerateRandomSignMatrix(64, 32)
	_, err = QuantizeTurboQuant([]float32{1.0, 2.0}, rot, qjl, 32, 4)
	if err == nil {
		t.Errorf("expected error on input size not multiple of blockSize")
	}

	// Error in PolarQuant from QuantizeTurboQuant
	_, err = QuantizeTurboQuant(make([]float32, 32), []float32{1.0}, qjl, 32, 4)
	if err == nil {
		t.Errorf("expected error from PolarQuant")
	}

	// Error in QJLTransform from QuantizeTurboQuant
	_, err = QuantizeTurboQuant(make([]float32, 32), rot, []float32{1.0}, 32, 4)
	if err == nil {
		t.Errorf("expected error from QJLTransform")
	}

	// Valid QuantizeTurboQuant and DequantizeTurboQuant
	input := make([]float32, 32)
	for i := range input {
		input[i] = float32(i) - 16.0
	}
	tqBytes, err := QuantizeTurboQuant(input, rot, qjl, 32, 4)
	if err != nil {
		t.Fatalf("QuantizeTurboQuant failed: %v", err)
	}

	deqTQ, err := DequantizeTurboQuant(tqBytes, rot, qjl, 32)
	if err != nil || len(deqTQ) != 32 {
		t.Fatalf("DequantizeTurboQuant failed: %v", err)
	}

	// QJLTransform with negative projections
	qNeg, _, err := QJLTransform([]float32{-1.0}, []float32{1.0}, 1, 1)
	if err != nil || len(qNeg) != 1 || qNeg[0] != -1 {
		t.Errorf("expected -1 from negative projection, got: %v", qNeg)
	}

	// setFloat32 and getFloat32
	buf := make([]byte, 4)
	setFloat32(buf, 12.34)
	if getFloat32(buf) != 12.34 {
		t.Errorf("getFloat32/setFloat32 mismatch")
	}
}

func TestTurboQuantMatricesCoverage(t *testing.T) {
	// rotation_matrix not F32
	f := &GGUFFile{
		Tensors: []*TensorInfo{
			{Name: "turboquant.rotation_matrix", Type: GGMLTypeF16, Dimensions: []uint64{32}, Data: make([]byte, 128)},
			{Name: "turboquant.qjl_matrix", Type: GGMLTypeF32, Dimensions: []uint64{32}, Data: make([]byte, 128)},
		},
	}
	_, _, err := f.GetTurboQuantMatrices()
	if err == nil || err.Error() != "turboquant.rotation_matrix must be F32" {
		t.Errorf("expected rotation_matrix must be F32 error, got: %v", err)
	}

	// qjl_matrix not F32
	f.Tensors[0].Type = GGMLTypeF32
	f.Tensors[1].Type = GGMLTypeF16
	_, _, err = f.GetTurboQuantMatrices()
	if err == nil || err.Error() != "turboquant.qjl_matrix must be F32" {
		t.Errorf("expected qjl_matrix must be F32 error, got: %v", err)
	}

	// success
	f.Tensors[1].Type = GGMLTypeF32
	rot, qjl, err := f.GetTurboQuantMatrices()
	if err != nil || len(rot) != 32 || len(qjl) != 32 {
		t.Errorf("expected success in GetTurboQuantMatrices, got err: %v", err)
	}

	// missing matrices
	f.Tensors = []*TensorInfo{}
	_, _, err = f.GetTurboQuantMatrices()
	if err == nil || err.Error() != "turboquant matrices not found in GGUF" {
		t.Errorf("expected turboquant matrices not found in GGUF, got: %v", err)
	}
}

func TestStructsCoverage(t *testing.T) {
	// SizeBytes for Q2_K, TQ2_0, MXFP4, default
	tQ2 := &TensorInfo{Type: GGMLTypeQ2_K, Dimensions: []uint64{256}}
	if tQ2.SizeBytes() != 84 {
		t.Errorf("expected 84 for Q2_K")
	}
	tTQ2 := &TensorInfo{Type: GGMLTypeTQ2_0, Dimensions: []uint64{256}}
	if tTQ2.SizeBytes() != 328 {
		t.Errorf("expected 328 for TQ2_0")
	}
	tMX := &TensorInfo{Type: GGMLTypeMXFP4, Dimensions: []uint64{32}}
	if tMX.SizeBytes() != 18 {
		t.Errorf("expected 18 for MXFP4")
	}
	tDef := &TensorInfo{Type: GGMLType(9999), Dimensions: []uint64{32}}
	if tDef.SizeBytes() != 0 {
		t.Errorf("expected 0 for unknown type")
	}

	// Error strings
	eMagic := ErrInvalidMagic{Magic: 0x1234}
	if eMagic.Error() == "" {
		t.Errorf("expected error string")
	}
	eVer := ErrUnsupportedVersion{Version: 99}
	if eVer.Error() == "" {
		t.Errorf("expected error string")
	}

	// String() for all GGMLTypes
	allTypes := []GGMLType{
		GGMLTypeF32, GGMLTypeF16, GGMLTypeQ4_0, GGMLTypeQ4_1, GGMLTypeQ5_0,
		GGMLTypeQ8_0, GGMLTypeQ2_K, GGMLTypeBF16, GGMLTypeQ3_K, GGMLTypeQ4_K,
		GGMLTypeQ5_K, GGMLTypeQ6_K, GGMLTypeQ8_K, GGMLTypeIQ2_XXS, GGMLTypeIQ2_XS,
		GGMLTypeIQ3_XXS, GGMLTypeIQ1_S, GGMLTypeIQ4_NL, GGMLTypeIQ3_S, GGMLTypeIQ2_S,
		GGMLTypeIQ4_XS, GGMLTypeIQ1_M, GGMLTypeMXFP4, GGMLTypeTQ1_0, GGMLTypeTQ2_0,
		GGMLTypeQ4_K_S, GGMLType(9999),
	}
	for _, typ := range allTypes {
		if typ.String() == "" {
			t.Errorf("expected non-empty string for type %d", typ)
		}
	}
}

func TestReaderCoverage(t *testing.T) {
	// LoadFile error paths
	// 1. File does not exist
	_, err := LoadFile("non_existent_file.gguf")
	if err == nil {
		t.Errorf("expected error for non_existent_file")
	}

	// Helper to write a temp file
	writeTemp := func(data []byte) string {
		tmp := filepath.Join(t.TempDir(), "test.gguf")
		if err := os.WriteFile(tmp, data, 0600); err != nil {
			t.Fatalf("failed to write temp file: %v", err)
		}
		return tmp
	}

	// 2. Empty file (0 bytes) -> mmap error
	tmpEmpty := writeTemp([]byte{})
	_, err = LoadFile(tmpEmpty)
	if err == nil {
		t.Errorf("expected error for empty file")
	}

	// 2b. fStatHook failure
	origStat := fStatHook
	fStatHook = func(f *os.File) (os.FileInfo, error) {
		return nil, os.ErrPermission
	}
	tmpStatErr := writeTemp([]byte{1, 2, 3})
	_, err = LoadFile(tmpStatErr)
	fStatHook = origStat
	if err == nil {
		t.Errorf("expected error on fStat failure")
	}

	// 2c. File too small (< 24 bytes)
	tmpTooSmall := writeTemp([]byte{1, 2, 3})
	_, err = LoadFile(tmpTooSmall)
	if err == nil {
		t.Errorf("expected error for file < 24 bytes")
	}

	// 3. Invalid magic
	badMagicBuf := make([]byte, 32)
	binary.LittleEndian.PutUint32(badMagicBuf[0:4], 0x11223344)
	tmpBadMagic := writeTemp(badMagicBuf)
	_, err = LoadFile(tmpBadMagic)
	if err == nil {
		t.Errorf("expected error for invalid magic")
	}

	// 4. Unsupported version
	unsupportedVerBuf := make([]byte, 32)
	binary.LittleEndian.PutUint32(unsupportedVerBuf[0:4], GGUFMagic)
	binary.LittleEndian.PutUint32(unsupportedVerBuf[4:8], 99)
	tmpUnsupp := writeTemp(unsupportedVerBuf)
	_, err = LoadFile(tmpUnsupp)
	if err == nil {
		t.Errorf("expected error for unsupported version")
	}

	// 5. Invalid KVCount (> 1000000)
	invKVBuf := make([]byte, 32)
	binary.LittleEndian.PutUint32(invKVBuf[0:4], GGUFMagic)
	binary.LittleEndian.PutUint32(invKVBuf[4:8], 3)
	binary.LittleEndian.PutUint64(invKVBuf[8:16], 0)
	binary.LittleEndian.PutUint64(invKVBuf[16:24], 2000000)
	tmpInvKV := writeTemp(invKVBuf)
	_, err = LoadFile(tmpInvKV)
	if err == nil {
		t.Errorf("expected error for invalid KVCount")
	}

	// 6. KVCount == 0 with tokenizer.ggml.tokens (ollama format)
	ollamaBuf := make([]byte, 512)
	binary.LittleEndian.PutUint32(ollamaBuf[0:4], GGUFMagic)
	binary.LittleEndian.PutUint32(ollamaBuf[4:8], 3)
	binary.LittleEndian.PutUint64(ollamaBuf[8:16], 0)
	binary.LittleEndian.PutUint64(ollamaBuf[16:24], 0) // KVCount = 0

	// Embed tokenizer.ggml.tokens at index 400
	copy(ollamaBuf[400:], []byte("tokenizer.ggml.tokens"))
	// Prefix with key 1: at index 300: length 4, "test", type uint32, value 42
	binary.LittleEndian.PutUint64(ollamaBuf[300:308], 4)
	copy(ollamaBuf[308:312], []byte("test"))
	binary.LittleEndian.PutUint32(ollamaBuf[312:316], uint32(GGUFMetadataValueTypeUint32))
	binary.LittleEndian.PutUint32(ollamaBuf[316:320], 42)

	// Prefix with key 2: at index 200: length 4, "subk", type uint32, value 99
	binary.LittleEndian.PutUint64(ollamaBuf[200:208], 4)
	copy(ollamaBuf[208:212], []byte("subk"))
	binary.LittleEndian.PutUint32(ollamaBuf[212:216], uint32(GGUFMetadataValueTypeUint32))
	binary.LittleEndian.PutUint32(ollamaBuf[216:220], 99)

	// Prefix with key 3: at index 100: length 4, "stop", type uint32, value 1
	binary.LittleEndian.PutUint64(ollamaBuf[100:108], 4)
	copy(ollamaBuf[108:112], []byte("stop"))
	binary.LittleEndian.PutUint32(ollamaBuf[112:116], uint32(GGUFMetadataValueTypeUint32))
	binary.LittleEndian.PutUint32(ollamaBuf[116:120], 1)

	tmpOllama := writeTemp(ollamaBuf)
	fOllama, err := LoadFile(tmpOllama)
	if err != nil {
		t.Fatalf("LoadFile ollama format failed: %v", err)
	}
	_ = fOllama.Close()

	// 6a. Ollama format where no key prefix is found (keyStart == 0 break)
	ollamaNoKeys := make([]byte, 512)
	binary.LittleEndian.PutUint32(ollamaNoKeys[0:4], GGUFMagic)
	binary.LittleEndian.PutUint32(ollamaNoKeys[4:8], 3)
	copy(ollamaNoKeys[350:], []byte("tokenizer.ggml.tokens"))
	tmpNoKeys := writeTemp(ollamaNoKeys)
	fNoKeys, err := LoadFile(tmpNoKeys)
	if err != nil {
		t.Fatalf("LoadFile ollama no keys failed: %v", err)
	}
	_ = fNoKeys.Close()

	// 6b. Ollama format break on keyLen extending beyond pos (keyStart+keyLen >= pos)
	ollamaBufOverflow := make([]byte, 512)
	binary.LittleEndian.PutUint32(ollamaBufOverflow[0:4], GGUFMagic)
	binary.LittleEndian.PutUint32(ollamaBufOverflow[4:8], 3)
	copy(ollamaBufOverflow[200:], []byte("tokenizer.ggml.tokens"))
	binary.LittleEndian.PutUint64(ollamaBufOverflow[150:158], 80) // keyLen 80 >= pos (200 - 158 = 42)
	tmpOllamaOverflow := writeTemp(ollamaBufOverflow)
	fOOB, err := LoadFile(tmpOllamaOverflow)
	if err == nil {
		_ = fOOB.Close()
	}

	// 6c. Ollama format break on readValue error (bad valType)
	ollamaBufBadVal := make([]byte, 512)
	binary.LittleEndian.PutUint32(ollamaBufBadVal[0:4], GGUFMagic)
	binary.LittleEndian.PutUint32(ollamaBufBadVal[4:8], 3)
	copy(ollamaBufBadVal[200:], []byte("tokenizer.ggml.tokens"))
	binary.LittleEndian.PutUint64(ollamaBufBadVal[150:158], 4)
	copy(ollamaBufBadVal[158:162], []byte("badt"))
	binary.LittleEndian.PutUint32(ollamaBufBadVal[162:166], 9999) // bad valType
	tmpOllamaBadVal := writeTemp(ollamaBufBadVal)
	fBadVal, err := LoadFile(tmpOllamaBadVal)
	if err == nil {
		_ = fBadVal.Close()
	}

	// 7. Standard format with metadata types, general.alignment (uint64 and float64), and tensors
	makeStandardGGUF := func(alignType GGUFMetadataValueType, tensorOffset uint64) []byte {
		buf := make([]byte, 2048)
		binary.LittleEndian.PutUint32(buf[0:4], GGUFMagic)
		binary.LittleEndian.PutUint32(buf[4:8], 3)
		binary.LittleEndian.PutUint64(buf[8:16], 1)  // 1 tensor
		binary.LittleEndian.PutUint64(buf[16:24], 2) // 2 KV pairs

		off := 24
		// KV 1: "general.alignment"
		key := "general.alignment"
		binary.LittleEndian.PutUint64(buf[off:off+8], uint64(len(key)))
		off += 8
		copy(buf[off:off+len(key)], key)
		off += len(key)
		binary.LittleEndian.PutUint32(buf[off:off+4], uint32(alignType))
		off += 4
		switch alignType {
		case GGUFMetadataValueTypeUint32:
			binary.LittleEndian.PutUint32(buf[off:off+4], 64)
			off += 4
		case GGUFMetadataValueTypeUint64:
			binary.LittleEndian.PutUint64(buf[off:off+8], 64)
			off += 8
		case GGUFMetadataValueTypeFloat64:
			binary.LittleEndian.PutUint64(buf[off:off+8], mathx.Float64bits(64.0))
			off += 8
		}

		// KV 2: "general.architecture" = "gemma4"
		key2 := "general.architecture"
		binary.LittleEndian.PutUint64(buf[off:off+8], uint64(len(key2)))
		off += 8
		copy(buf[off:off+len(key2)], key2)
		off += len(key2)
		binary.LittleEndian.PutUint32(buf[off:off+4], uint32(GGUFMetadataValueTypeString))
		off += 4
		valStr := "gemma4"
		binary.LittleEndian.PutUint64(buf[off:off+8], uint64(len(valStr)))
		off += 8
		copy(buf[off:off+len(valStr)], valStr)
		off += len(valStr)

		// Tensor info: name "weight", dims 1, [32], type F32, offset tensorOffset
		tName := "weight"
		binary.LittleEndian.PutUint64(buf[off:off+8], uint64(len(tName)))
		off += 8
		copy(buf[off:off+len(tName)], tName)
		off += len(tName)
		binary.LittleEndian.PutUint32(buf[off:off+4], 1) // 1 dim
		off += 4
		binary.LittleEndian.PutUint64(buf[off:off+8], 32) // dim 32
		off += 8
		binary.LittleEndian.PutUint32(buf[off:off+4], uint32(GGMLTypeF32))
		off += 4
		binary.LittleEndian.PutUint64(buf[off:off+8], tensorOffset)

		return buf
	}

	// Test general.alignment with uint64
	tmpU64 := writeTemp(makeStandardGGUF(GGUFMetadataValueTypeUint64, 0))
	fU64, err := LoadFile(tmpU64)
	if err != nil {
		t.Fatalf("LoadFile uint64 align failed: %v", err)
	}
	_ = fU64.Close()

	// Test general.alignment with float64
	tmpF64 := writeTemp(makeStandardGGUF(GGUFMetadataValueTypeFloat64, 0))
	fF64, err := LoadFile(tmpF64)
	if err != nil {
		t.Fatalf("LoadFile float64 align failed: %v", err)
	}
	_ = fF64.Close()

	// Test tensor offset out of bounds
	tmpOOB := writeTemp(makeStandardGGUF(GGUFMetadataValueTypeUint32, 100000))
	_, err = LoadFile(tmpOOB)
	if err == nil {
		t.Errorf("expected error on tensor offset out of bounds")
	}

	// Test readValue error in standard format KV parsing
	bufBadKV := makeStandardGGUF(GGUFMetadataValueTypeUint32, 0)
	// Corrupt valType of KV 2 to 9999 (offset 24 + 8+17+4+4 + 8+20 = 85)
	binary.LittleEndian.PutUint32(bufBadKV[85:89], 9999)
	tmpBadKV := writeTemp(bufBadKV)
	_, err = LoadFile(tmpBadKV)
	if err == nil {
		t.Errorf("expected error on bad KV valType")
	}

	// Test readString error in tensor parsing
	bufBadTensor := makeStandardGGUF(GGUFMetadataValueTypeUint32, 0)
	// Truncate file right at tensor name
	tmpBadTensor := writeTemp(bufBadTensor[:110])
	_, err = LoadFile(tmpBadTensor)
	if err == nil {
		t.Errorf("expected error on truncated tensor header")
	}

	// Test readValue all types
	testData := make([]byte, 1024)
	testData[0] = 5                                    // uint8
	testData[1] = 0xFE                                 // int8 (-2)
	binary.LittleEndian.PutUint16(testData[2:4], 1000) // uint16
	var i16Val int16 = -1000
	binary.LittleEndian.PutUint16(testData[4:6], uint16(i16Val)) // int16
	binary.LittleEndian.PutUint32(testData[6:10], 100000)        // uint32
	var i32Val int32 = -100
	binary.LittleEndian.PutUint32(testData[10:14], uint32(i32Val)) // int32
	binary.LittleEndian.PutUint32(testData[14:18], mathx.Float32bits(3.14))
	testData[18] = 1 // bool true
	// Array: type Uint32, len 2, elements [10, 20]
	binary.LittleEndian.PutUint32(testData[19:23], uint32(GGUFMetadataValueTypeUint32))
	binary.LittleEndian.PutUint64(testData[23:31], 2)
	binary.LittleEndian.PutUint32(testData[31:35], 10)
	binary.LittleEndian.PutUint32(testData[35:39], 20)
	binary.LittleEndian.PutUint64(testData[39:47], 999999) // uint64
	var i64Val int64 = -999999
	binary.LittleEndian.PutUint64(testData[47:55], uint64(i64Val))           // int64
	binary.LittleEndian.PutUint64(testData[55:63], mathx.Float64bits(2.718)) // float64

	v, _, _ := readValue(testData, 0, GGUFMetadataValueTypeUint8)
	if v != uint8(5) {
		t.Errorf("uint8 mismatch")
	}
	v, _, _ = readValue(testData, 1, GGUFMetadataValueTypeInt8)
	if v != int8(-2) {
		t.Errorf("int8 mismatch")
	}
	v, _, _ = readValue(testData, 2, GGUFMetadataValueTypeUint16)
	if v != uint16(1000) {
		t.Errorf("uint16 mismatch")
	}
	v, _, _ = readValue(testData, 4, GGUFMetadataValueTypeInt16)
	if v != int16(-1000) {
		t.Errorf("int16 mismatch")
	}
	v, _, _ = readValue(testData, 6, GGUFMetadataValueTypeUint32)
	if v != uint32(100000) {
		t.Errorf("uint32 mismatch")
	}
	v, _, _ = readValue(testData, 10, GGUFMetadataValueTypeInt32)
	if v != int32(-100) {
		t.Errorf("int32 mismatch")
	}
	v, _, _ = readValue(testData, 14, GGUFMetadataValueTypeFloat32)
	if v != float32(3.14) {
		t.Errorf("float32 mismatch")
	}
	v, _, _ = readValue(testData, 18, GGUFMetadataValueTypeBool)
	if v != true {
		t.Errorf("bool mismatch")
	}
	v, _, _ = readValue(testData, 19, GGUFMetadataValueTypeArray)
	arr := v.([]interface{})
	if len(arr) != 2 || arr[0].(uint32) != 10 || arr[1].(uint32) != 20 {
		t.Errorf("array mismatch: %v", arr)
	}
	v, _, _ = readValue(testData, 39, GGUFMetadataValueTypeUint64)
	if v != uint64(999999) {
		t.Errorf("uint64 mismatch")
	}
	v, _, _ = readValue(testData, 47, GGUFMetadataValueTypeInt64)
	if v != int64(-999999) {
		t.Errorf("int64 mismatch")
	}
	v, _, _ = readValue(testData, 55, GGUFMetadataValueTypeFloat64)
	if v != 2.718 {
		t.Errorf("float64 mismatch")
	}
	// Unknown metadata type
	_, _, err = readValue(testData, 0, GGUFMetadataValueType(9999))
	if err == nil {
		t.Errorf("expected error on unknown metadata type")
	}

	// Array with invalid element type error
	badArrData := make([]byte, 32)
	binary.LittleEndian.PutUint32(badArrData[0:4], 9999) // bad element type
	binary.LittleEndian.PutUint64(badArrData[4:12], 1)   // len 1
	_, _, err = readValue(badArrData, 0, GGUFMetadataValueTypeArray)
	if err == nil {
		t.Errorf("expected error in array reading with bad element type")
	}

	// readString errors: offset+8 > len(data) and offset+8+length > len(data)
	_, _, err = readString([]byte{1, 2}, 0)
	if err == nil {
		t.Errorf("expected error on short string header")
	}
	strShort := make([]byte, 10)
	binary.LittleEndian.PutUint64(strShort[0:8], 100) // len 100, but buffer is 10
	_, _, err = readString(strShort, 0)
	if err == nil {
		t.Errorf("expected error on truncated string body")
	}

	// GetGapTensors with gap > 1MB
	fGap := &GGUFFile{
		Data:       make([]byte, 3*1024*1024),
		DataOffset: 0,
		Tensors: []*TensorInfo{
			{Name: "t1", Offset: 0, Type: GGMLTypeF32, Dimensions: []uint64{32}}, // 128 bytes
			{Name: "t2", Offset: 2 * 1024 * 1024, Type: GGMLTypeF32, Dimensions: []uint64{32}},
		},
	}
	gaps := fGap.GetGapTensors()
	if len(gaps) != 1 {
		t.Errorf("expected 1 gap, got %d", len(gaps))
	}
}
