//go:build webui

package config

import (
	"os"
	"strconv"
	"strings"
)

// Environment variables understood by the WebUI. Flags take precedence over
// the environment; the environment takes precedence over the built-in
// defaults.
//
// These existed in Dockerfile.webui and docker-compose.webui.yml but were
// never read by the binary, which meant setting WEBUI_API_KEY silently
// produced an unauthenticated server.
const (
	EnvPort           = "WEBUI_PORT"
	EnvMetricsPort    = "WEBUI_METRICS_PORT"
	EnvHost           = "WEBUI_HOST"
	EnvAPIKey         = "WEBUI_API_KEY"
	EnvAllowedOrigins = "WEBUI_ALLOWED_ORIGINS"
	EnvModelPath      = "WEBUI_MODEL_PATH"
	EnvDataDir        = "WEBUI_DATA_DIR"
	EnvCacheDir       = "WEBUI_CACHE_DIR"
)

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

// ApplyEnv overlays environment variables onto the config. Only variables
// that are actually set are applied, so an unset variable never clobbers an
// explicit flag value.
func (c *Config) ApplyEnv() {
	if v := os.Getenv(EnvPort); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			c.Port = n
		}
	}
	if v := os.Getenv(EnvMetricsPort); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			c.MetricsPort = n
		}
	}
	if v := os.Getenv(EnvHost); v != "" {
		c.Host = v
	}
	if v := os.Getenv(EnvAPIKey); v != "" {
		c.APIKey = v
	}
	if v := os.Getenv(EnvAllowedOrigins); v != "" {
		c.AllowedOrigins = SplitOrigins(v)
	}
	if v := os.Getenv(EnvModelPath); v != "" {
		c.ModelPath = v
	}
}

// SplitOrigins parses a comma-separated origin list. It always returns a
// non-nil slice so the result marshals to a JSON array rather than null.
func SplitOrigins(origins string) []string {
	result := []string{}
	if origins == "" {
		return result
	}
	for _, origin := range strings.Split(origins, ",") {
		if trimmed := strings.TrimSpace(origin); trimmed != "" {
			result = append(result, trimmed)
		}
	}
	return result
}
