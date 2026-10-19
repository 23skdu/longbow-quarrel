package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

func TestEngine_CPUKVCache_And_MathHelpers(t *testing.T) {
	// 1. Sliding window CPU KV cache
	cache := NewCPUKVCacheWithWindow(4, 128)
	if cache == nil || cache.MaxLen != 128 {
		t.Fatalf("expected cache with maxLen 128")
	}
	cache.Keys[0] = append(cache.Keys[0], 1.0, 2.0)
	cache.Values[0] = append(cache.Values[0], 3.0, 4.0)
	cache.Reset()
	if len(cache.Keys[0]) != 0 || len(cache.Values[0]) != 0 {
		t.Errorf("expected Reset to clear keys and values")
	}
	var nilCache *CPUKVCache
	nilCache.Reset() // should not panic

	// 2. Math activation functions in cpu_weights.go
	if s := siluCPU(-25); s != 0 {
		t.Errorf("siluCPU(-25) got %f, want 0", s)
	}
	if s := siluCPU(25); s != 25 {
		t.Errorf("siluCPU(25) got %f, want 25", s)
	}
	if s := siluCPU(0); s != 0 {
		t.Errorf("siluCPU(0) got %f, want 0", s)
	}

	if sp := softplusCPU(25); sp != 25 {
		t.Errorf("softplusCPU(25) got %f, want 25", sp)
	}
	if sp := softplusCPU(-25); sp != 0 {
		t.Errorf("softplusCPU(-25) got %f, want 0", sp)
	}
	_ = softplusCPU(0)

	if sig := sigmoidCPU(25); sig != 1.0 {
		t.Errorf("sigmoidCPU(25) got %f, want 1.0", sig)
	}
	if sig := sigmoidCPU(-25); sig != 0.0 {
		t.Errorf("sigmoidCPU(-25) got %f, want 0.0", sig)
	}
	_ = sigmoidCPU(0)

	// partialRoPECPU edge cases
	partialRoPECPU(nil, nil, 1, 16, 0, 10000)

	// CPULayerScratch
	scDefault := NewCPULayerScratch(0, 0, 0)
	if len(scDefault.Normed) != 4096 {
		t.Errorf("expected 4096 default dim, got %d", len(scDefault.Normed))
	}
	scCustom := NewCPULayerScratch(512, 8, 64)
	if len(scCustom.Normed) != 512 {
		t.Errorf("expected 512 dim, got %d", len(scCustom.Normed))
	}
	res := ApplyLayerCPUKVWithScratch(nil, []float32{1.0}, 0, 0, nil, config.Config{}, scCustom)
	if len(res) != 1 || res[0] != 1.0 {
		t.Errorf("expected identity on nil weights")
	}
}

func TestEngine_BatchManager_Preempt(t *testing.T) {
	cm := NewContinuousBatchManager()
	if evicted := cm.PreemptLowestPriority(nil); evicted != 0 {
		t.Errorf("expected 0 evicted on empty batch, got %d", evicted)
	}

	cm.running[101] = &Sequence{ID: 101, Priority: PriorityLow}
	cm.running[102] = &Sequence{ID: 102, Priority: PriorityNormal}
	cm.running[103] = &Sequence{ID: 103, Priority: PriorityUrgent}

	evicted := cm.PreemptLowestPriority(nil)
	if evicted != 101 {
		t.Errorf("expected sequence 101 evicted, got %d", evicted)
	}

	// Now remaining are 102 and 103. Preempt again -> 102
	evicted = cm.PreemptLowestPriority(nil)
	if evicted != 102 {
		t.Errorf("expected sequence 102 evicted, got %d", evicted)
	}

	// Only 103 (PriorityUrgent) remains -> should not preempt urgent
	evicted = cm.PreemptLowestPriority(nil)
	if evicted != 0 {
		t.Errorf("expected 0 evicted when only urgent left, got %d", evicted)
	}
}

func TestEngine_Types_FreeSafety(t *testing.T) {
	// Weights Free() safety
	var lw LlamaWeights
	lw.Free()

	var mw MambaWeights
	mw.Free()

	var ms MambaState
	ms.Free()

	var mew MOEExpertWeights
	mew.Free()

	var msw MOESharedWeights
	msw.Free()

	var mrw MOERouterWeights
	mrw.Free()

	var mlw MOELayerWeights
	mlw.Free()
}
