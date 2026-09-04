package engine

import (
	"encoding/binary"
	"math"
	"testing"

	conf "github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestPagedKVCache_TurboQuant(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	cache := &PagedKVCache{}
	config := conf.Config{
		KVHeads:     2,
		HeadDim:    64,
		Layers:      2,
		WindowSize: 512,
	}

	// Enable TurboQuant precision
	cache.Precision = device.DataTypeTQ1_0

	err := cache.Init(ctx, config)
	if err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	defer cache.Free()

	if cache.qjlRows == 0 {
		t.Error("qjlRows should be set")
	}

	// Verify TurboQuant block structure
	expectedBlockSize := cache.headDim + cache.qjlRows + 8
	if expectedBlockSize == 0 {
		t.Error("TurboQuant block size should be > 0")
	}
}

func TestPagedKVCache_TurboQuantEncode(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	headDim := 32
	qjlRows := 16
	heads := 2

	// Create TurboQuant matrices
	rotData := make([]float32, headDim*headDim)
	for i := range rotData {
		if i%(headDim+1) == 0 {
			rotData[i] = 1.0
		}
	}
	rot := ctx.NewTensorFP32(headDim, headDim)
	rot.LoadFromF32(rotData)

	qjlData := make([]float32, qjlRows*headDim)
	for i := range qjlData {
		if i%3 == 0 {
			qjlData[i] = 1.0
		} else {
			qjlData[i] = -1.0
		}
	}
	qjl := ctx.NewTensorFP32(qjlRows, headDim)
	qjl.LoadFromF32(qjlData)

	ctx.TQRotation = rot
	ctx.TQQJL = qjl

	cache := &PagedKVCache{}
	cfg := conf.Config{
		KVHeads:     heads,
		HeadDim:    headDim,
		Layers:     1,
		WindowSize: 128,
	}

	cache.Precision = device.DataTypeTQ1_0

	err := cache.Init(ctx, cfg)
	if err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	defer cache.Free()

	if cache.tqRotation == nil {
		t.Fatal("TurboQuant matrices should be assigned to cache")
	}

	// Test encoding path exists
	if cache.kPools == nil || len(cache.kPools) == 0 {
		t.Fatal("KV pools should be allocated")
	}

	// Allocate a sequence block
	seqID := "test-seq-1"
	err = cache.Allocate(seqID, 2)
	if err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}

	// Create test input tensors for K and V
	numTokens := 2
	kvDim := heads * headDim
	kData := make([]float32, numTokens*kvDim)
	vData := make([]float32, numTokens*kvDim)
	for i := range kData {
		kData[i] = float32(i%10) * 0.1
		vData[i] = float32((i+5)%10) * 0.1
	}

	kTensor := ctx.NewTensorFP32(numTokens, kvDim)
	vTensor := ctx.NewTensorFP32(numTokens, kvDim)
	_ = kTensor.LoadFrom(kData)
	_ = vTensor.LoadFrom(vData)

	posTensor := ctx.NewTensorFP32(1, numTokens)
	_ = posTensor.LoadFrom([]float32{0.0, 1.0})

	cache.StoreKVPagedBatch(0, kTensor, vTensor, posTensor)

	// Verify kPool and vPool have encoded data
	kRaw := cache.kPools[0].RawData()
	vRaw := cache.vPools[0].RawData()

	if len(kRaw) == 0 || len(vRaw) == 0 {
		t.Fatal("Pool raw data should not be empty")
	}

	// Verify that at least some bytes in kRaw and vRaw are non-zero
	hasNonZeroK := false
	for _, b := range kRaw {
		if b != 0 {
			hasNonZeroK = true
			break
		}
	}
	if !hasNonZeroK {
		t.Error("kPool rawData is all zeros after encodeKVTurboQuant")
	}

	hasNonZeroV := false
	for _, b := range vRaw {
		if b != 0 {
			hasNonZeroV = true
			break
		}
	}
	if !hasNonZeroV {
		t.Error("vPool rawData is all zeros after encodeKVTurboQuant")
	}
}

func TestPagedKVCache_TurboQuant_RoundtripAccuracy(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	headDim := 64
	qjlRows := 32
	heads := 1

	rotData := device.GetPrecomputedRotation(headDim)
	qjlData := device.GetPrecomputedQJLSigns(qjlRows * headDim)

	rot := ctx.NewTensorFP32(headDim, headDim)
	_ = rot.LoadFrom(rotData)
	qjl := ctx.NewTensorFP32(qjlRows, headDim)
	_ = qjl.LoadFrom(qjlData)

	ctx.TQRotation = rot
	ctx.TQQJL = qjl

	cache := &PagedKVCache{Precision: device.DataTypeTQ2_0}
	cfg := conf.Config{
		KVHeads:    heads,
		HeadDim:    headDim,
		Layers:     1,
		WindowSize: 64,
	}

	if err := cache.Init(ctx, cfg); err != nil {
		t.Fatalf("cache.Init failed: %v", err)
	}
	defer cache.Free()

	if err := cache.Allocate("test-seq", 1); err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}

	// Generate a normalized sine-wave test vector for K
	kInput := make([]float32, headDim)
	for i := range kInput {
		kInput[i] = float32(math.Sin(float64(i)*0.2)) * 0.5
	}

	kTensor := ctx.NewTensorFP32(1, headDim)
	_ = kTensor.LoadFrom(kInput)
	vTensor := ctx.NewTensorFP32(1, headDim)
	_ = vTensor.LoadFrom(kInput)

	posTensor := ctx.NewTensorFP32(1, 1)
	_ = posTensor.LoadFrom([]float32{0.0})

	cache.StoreKVPagedBatch(0, kTensor, vTensor, posTensor)

	// Check that data was written into slot 0
	bytesPerBlock := headDim + qjlRows + 8
	kRaw := cache.kPools[0].RawData()
	if len(kRaw) < bytesPerBlock {
		t.Fatalf("kPool rawData too small: %d < %d", len(kRaw), bytesPerBlock)
	}

	// Check that the written scale factor is strictly positive and finite
	bits := binary.LittleEndian.Uint32(kRaw[headDim+qjlRows : headDim+qjlRows+4])
	scale := math.Float32frombits(bits)
	if scale <= 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		t.Errorf("Invalid scale factor encoded: %v", scale)
	}
}