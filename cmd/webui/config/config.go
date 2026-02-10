//go:build webui

package config

type Config struct {
	Port           int
	MetricsPort    int
	Host           string
	ModelPath      string
	MaxTokens      int
	Temperature    float64
	TopK           int
	TopP           float64
	APIKey         string
	AllowedOrigins []string
}

func DefaultConfig() Config {
	return Config{
		Port:        8080,
		MetricsPort: 9090,
		Host:        "0.0.0.0",
		ModelPath:   "",
		MaxTokens:   1024,
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.95,
	}
}
