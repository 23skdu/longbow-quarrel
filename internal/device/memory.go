package device

// memory.go provides platform-agnostic memory configuration and constants.
// This file is compiled into all builds and defines defaults for all backends.

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
)

const (
	DefaultMaxMemoryMetal int64 = 32 * 1024 * 1024 * 1024
	DefaultMaxMemoryCUDA  int64 = 8 * 1024 * 1024 * 1024
)

type MemoryConfig struct {
	MaxMemory int64
}

var defaultMemoryConfig = &MemoryConfig{
	MaxMemory: 0,
}

func GetMemoryConfig() *MemoryConfig {
	if atomic.LoadInt64(&defaultMemoryConfig.MaxMemory) == 0 {
		atomic.StoreInt64(&defaultMemoryConfig.MaxMemory, DefaultMaxMemoryCUDA)
	}
	return defaultMemoryConfig
}

func SetMaxMemory(maxMemory int64) {
	atomic.StoreInt64(&defaultMemoryConfig.MaxMemory, maxMemory)
}

var maxMemoryMB int64

func SetMaxMemoryMB(mb int64) {
	atomic.StoreInt64(&maxMemoryMB, mb)
	SetMaxMemory(mb * 1024 * 1024)
}

func GetMaxMemoryMB() int64 {
	return atomic.LoadInt64(&maxMemoryMB)
}

// ===== Proactive Memory Governor (v0.4.0 Part 10) =====

const DefaultMemoryPressureThreshold = 0.85

type MemoryGovernor struct {
	mu             sync.Mutex
	defragHandlers []func()
	offloadHandlers []func()
	customRAMUsage  func() (used, total int64, pct float64)
	customVRAMUsage func() (used, total int64, pct float64)
}

var globalGovernor = &MemoryGovernor{
	defragHandlers:  make([]func(), 0),
	offloadHandlers: make([]func(), 0),
}

// GetMemoryGovernor returns the global memory governor singleton.
func GetMemoryGovernor() *MemoryGovernor {
	return globalGovernor
}

// RegisterDefragHandler registers a callback invoked when memory pressure exceeds threshold.
func (g *MemoryGovernor) RegisterDefragHandler(h func()) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.defragHandlers = append(g.defragHandlers, h)
}

// RegisterOffloadHandler registers a callback invoked when severe pressure occurs.
func (g *MemoryGovernor) RegisterOffloadHandler(h func()) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.offloadHandlers = append(g.offloadHandlers, h)
}

// SetCustomMemorySamplers allows injecting test samplers for RAM/VRAM.
func (g *MemoryGovernor) SetCustomMemorySamplers(
	ramSampler func() (used, total int64, pct float64),
	vramSampler func() (used, total int64, pct float64),
) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.customRAMUsage = ramSampler
	g.customVRAMUsage = vramSampler
}

// GetHostMemoryUsage returns the host RAM utilization.
func GetHostMemoryUsage() (usedBytes, totalBytes int64, pct float64) {
	if globalGovernor.customRAMUsage != nil {
		return globalGovernor.customRAMUsage()
	}

	// Try reading /proc/meminfo on Linux
	data, err := os.ReadFile("/proc/meminfo")
	if err == nil {
		lines := strings.Split(string(data), "\n")
		var memTotal, memAvailable int64
		for _, line := range lines {
			if strings.HasPrefix(line, "MemTotal:") {
				_, _ = fmt.Sscanf(line, "MemTotal: %d kB", &memTotal)
			} else if strings.HasPrefix(line, "MemAvailable:") {
				_, _ = fmt.Sscanf(line, "MemAvailable: %d kB", &memAvailable)
			}
		}
		if memTotal > 0 {
			totalBytes = memTotal * 1024
			availBytes := memAvailable * 1024
			usedBytes = totalBytes - availBytes
			pct = float64(usedBytes) / float64(totalBytes)
			return usedBytes, totalBytes, pct
		}
	}

	// Fallback to Go runtime memstats
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	usedBytes = int64(m.Alloc) // #nosec G115
	totalBytes = GetMemoryConfig().MaxMemory
	if totalBytes > 0 {
		pct = float64(usedBytes) / float64(totalBytes)
	}
	return usedBytes, totalBytes, pct
}

// CheckMemoryPressure checks if host RAM or device VRAM exceeds the threshold.
func (g *MemoryGovernor) CheckMemoryPressure(threshold float64) (ramExceeded, vramExceeded bool, ramPct, vramPct float64) {
	_, _, ramPct = GetHostMemoryUsage()
	if g.customVRAMUsage != nil {
		_, _, vramPct = g.customVRAMUsage()
	} else {
		// Use allocated bytes vs MaxMemory
		alloc := AllocatedBytes()
		maxMem := GetMemoryConfig().MaxMemory
		if maxMem > 0 {
			vramPct = float64(alloc) / float64(maxMem)
		}
	}

	if ramPct >= threshold {
		ramExceeded = true
	}
	if vramPct >= threshold {
		vramExceeded = true
	}
	return ramExceeded, vramExceeded, ramPct, vramPct
}

// TriggerGovernor runs proactive defragmentation and offloading if memory pressure exceeds threshold.
func (g *MemoryGovernor) TriggerGovernor(threshold float64) (actionTaken string) {
	ramExceeded, vramExceeded, ramPct, vramPct := g.CheckMemoryPressure(threshold)
	if !ramExceeded && !vramExceeded {
		return "none"
	}

	g.mu.Lock()
	defragCopy := make([]func(), len(g.defragHandlers))
	copy(defragCopy, g.defragHandlers)
	offloadCopy := make([]func(), len(g.offloadHandlers))
	copy(offloadCopy, g.offloadHandlers)
	g.mu.Unlock()

	// Execute defragmentation
	for _, h := range defragCopy {
		if h != nil {
			h()
		}
	}
	actionTaken = "defrag"

	// If extreme pressure (>92%), trigger page offload
	if ramPct > 0.92 || vramPct > 0.92 {
		for _, h := range offloadCopy {
			if h != nil {
				h()
			}
		}
		actionTaken = "defrag_and_offload"
	}

	return actionTaken
}

