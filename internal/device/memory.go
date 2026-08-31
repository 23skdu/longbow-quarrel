package device

// memory.go provides platform-agnostic memory configuration and constants.
// This file is compiled into all builds and defines defaults for all backends.

import "sync/atomic"

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
