//go:build linux && amd64 && cuda && cgo

package engine

import (
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

// TestCUDAEngineWithAQuantizedKVCache runs the GPU engine with the
// quantized KV cache and compares its logits with the run that uses the
// F16 pool.
//
// The quantized cache is the one the roadmap calls "implemented and never
// enabled": the pool was allocated with four bytes per element instead of
// one, the ordinary F16 store filled it, and the quantized attention read
// it back as int8. Nothing ever set the precision, so no binary could
// reach it.
func TestCUDAEngineWithAQuantizedKVCache(t *testing.T) {
	dir := t.TempDir()
	path := buildTestGGUF(t, dir)

	base := config.Default()
	base.KVCacheSize = 128
	base.SeqLen = 128
	base.NumGPULayers = -1

	for _, tc := range []struct {
		name      string
		kvType    config.KVCacheType
		tolerance float64
	}{
		{name: "q8_0", kvType: config.KVCacheQ8_0, tolerance: 0.05},
		{name: "fp8", kvType: config.KVCacheFP8, tolerance: 0.2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The F16 run is the reference.
			f16Cfg := base
			f16Cfg.KVCacheType = config.KVCacheF16
			f16Created, err := NewcudaEngine(path, f16Cfg)
			if err != nil {
				t.Fatalf("failed to create the F16 engine: %v", err)
			}
			defer f16Created.Close()

			quantCfg := base
			quantCfg.KVCacheType = tc.kvType
			quantCreated, err := NewcudaEngine(path, quantCfg)
			if err != nil {
				t.Fatalf("failed to create the %s engine: %v", tc.name, err)
			}
			defer quantCreated.Close()

			quantEngine := quantCreated.(*cudaEngine)
			if quantEngine.cache.Precision == device.DataTypeF16 {
				t.Fatalf("the %s cache was not selected: the precision is %v", tc.name, quantEngine.cache.Precision)
			}

			batch := &BatchDescriptor{
				Sequences:   []*Sequence{{ID: 1, Tokens: []int{1, 2}, PromptLen: 2}},
				Tokens:      []int{1, 2},
				Offsets:     []int{0},
				ContextLens: []int{0},
				TokenToSeq:  []int{0, 0},
			}

			f16Batch, err := f16Created.(*cudaEngine).ForwardBatch(batch)
			if err != nil {
				t.Fatalf("the F16 forward failed: %v", err)
			}
			defer f16Batch[0].Free()

			quantBatch, err := quantEngine.ForwardBatch(batch)
			if err != nil {
				t.Fatalf("the %s forward failed: %v", tc.name, err)
			}
			defer quantBatch[0].Free()

			f16Logits := f16Batch[0].ToHostF32()
			quantLogits := quantBatch[0].ToHostF32()
			if len(f16Logits) != len(quantLogits) {
				t.Fatalf("the F16 run produced %d logits and the %s run %d",
					len(f16Logits), tc.name, len(quantLogits))
			}

			scale := 0.0
			for _, v := range f16Logits {
				if a := math.Abs(float64(v)); a > scale {
					scale = a
				}
			}
			if scale == 0 {
				scale = 1
			}

			for i := range f16Logits {
				f16Value := float64(f16Logits[i])
				quantValue := float64(quantLogits[i])
				if math.IsNaN(quantValue) {
					t.Fatalf("the logit %d is %v with the %s cache", i, quantLogits[i], tc.name)
				}
				diff := math.Abs(f16Value - quantValue)
				if diff > scale*tc.tolerance {
					t.Errorf("the logit %d is %v with the F16 cache and %v with the %s cache (difference %v)",
						i, f16Logits[i], quantLogits[i], tc.name, diff)
				}
			}
		})
	}
}
