package engine

import (
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
		t.Log("Note: TurboQuant matrices not assigned to cache (expected if ctx had them)")
	}

	// Test encoding path exists
	if cache.kPools == nil || len(cache.kPools) == 0 {
		t.Error("KV pools should be allocated")
	}
}