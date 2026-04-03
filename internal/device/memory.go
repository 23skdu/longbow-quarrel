package device

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
	if defaultMemoryConfig.MaxMemory == 0 {
		defaultMemoryConfig.MaxMemory = DefaultMaxMemoryCUDA
	}
	return defaultMemoryConfig
}

func SetMaxMemory(maxMemory int64) {
	defaultMemoryConfig.MaxMemory = maxMemory
}
