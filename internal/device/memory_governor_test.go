package device

import (
	"testing"
)

func TestMemoryConfig_GetAndSet(t *testing.T) {
	oldMax := GetMemoryConfig().MaxMemory
	defer SetMaxMemory(oldMax)

	SetMaxMemory(16 * 1024 * 1024 * 1024)
	if GetMemoryConfig().MaxMemory != 16*1024*1024*1024 {
		t.Errorf("expected 16GB, got %d", GetMemoryConfig().MaxMemory)
	}

	SetMaxMemoryMB(4096)
	if GetMaxMemoryMB() != 4096 {
		t.Errorf("expected 4096 MB, got %d", GetMaxMemoryMB())
	}
	if GetMemoryConfig().MaxMemory != 4096*1024*1024 {
		t.Errorf("expected 4096MB in bytes, got %d", GetMemoryConfig().MaxMemory)
	}
}

func TestMemoryGovernor_Trigger(t *testing.T) {
	gov := GetMemoryGovernor()

	defragCalled := 0
	offloadCalled := 0

	gov.RegisterDefragHandler(func() {
		defragCalled++
	})
	gov.RegisterOffloadHandler(func() {
		offloadCalled++
	})

	// 1. Normal conditions (<85%) -> action should be "none"
	gov.SetCustomMemorySamplers(
		func() (int64, int64, float64) {
			return 500, 1000, 0.50
		},
		func() (int64, int64, float64) {
			return 500, 1000, 0.50
		},
	)

	action := gov.TriggerGovernor(0.85)
	if action != "none" {
		t.Errorf("expected 'none', got %s", action)
	}
	if defragCalled != 0 || offloadCalled != 0 {
		t.Errorf("expected 0 callbacks called, got defrag=%d offload=%d", defragCalled, offloadCalled)
	}

	// 2. Memory pressure breached (e.g. 88%) -> action should be "defrag"
	gov.SetCustomMemorySamplers(
		func() (int64, int64, float64) {
			return 880, 1000, 0.88
		},
		func() (int64, int64, float64) {
			return 700, 1000, 0.70
		},
	)

	action = gov.TriggerGovernor(0.85)
	if action != "defrag" {
		t.Errorf("expected 'defrag', got %s", action)
	}
	if defragCalled != 1 || offloadCalled != 0 {
		t.Errorf("expected defrag=1 offload=0, got defrag=%d offload=%d", defragCalled, offloadCalled)
	}

	// 3. Extreme memory pressure breached (>92%) -> action should be "defrag_and_offload"
	gov.SetCustomMemorySamplers(
		func() (int64, int64, float64) {
			return 950, 1000, 0.95
		},
		func() (int64, int64, float64) {
			return 950, 1000, 0.95
		},
	)

	action = gov.TriggerGovernor(0.85)
	if action != "defrag_and_offload" {
		t.Errorf("expected 'defrag_and_offload', got %s", action)
	}
	if defragCalled != 2 || offloadCalled != 1 {
		t.Errorf("expected defrag=2 offload=1, got defrag=%d offload=%d", defragCalled, offloadCalled)
	}

	// Reset samplers to nil to test fallback to actual system readings
	gov.SetCustomMemorySamplers(nil, nil)
	used, total, pct := GetHostMemoryUsage()
	if total <= 0 && used <= 0 && pct < 0 {
		t.Errorf("invalid host memory usage: used=%d, total=%d, pct=%f", used, total, pct)
	}
}
