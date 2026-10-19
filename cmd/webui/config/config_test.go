//go:build webui

package config

import (
	"os"
	"reflect"
	"testing"
)

// unsetenv removes a variable for the duration of the test. t.Setenv records
// the original value and restores it on cleanup, so unsetting after t.Setenv
// keeps the test isolated.
func unsetenv(key string) error {
	return os.Unsetenv(key)
}

func TestApplyEnv_ReadsAllVariables(t *testing.T) {
	t.Setenv(EnvPort, "9999")
	t.Setenv(EnvMetricsPort, "9998")
	t.Setenv(EnvHost, "127.0.0.1")
	t.Setenv(EnvAPIKey, "secret-key")
	t.Setenv(EnvAllowedOrigins, "http://a.example, http://b.example")
	t.Setenv(EnvModelPath, "/models/test.gguf")

	c := DefaultConfig()
	c.ApplyEnv()

	if c.Port != 9999 {
		t.Errorf("expected port 9999, got %d", c.Port)
	}
	if c.MetricsPort != 9998 {
		t.Errorf("expected metrics port 9998, got %d", c.MetricsPort)
	}
	if c.Host != "127.0.0.1" {
		t.Errorf("expected host 127.0.0.1, got %q", c.Host)
	}
	if c.APIKey != "secret-key" {
		t.Errorf("expected api key 'secret-key', got %q", c.APIKey)
	}
	if c.ModelPath != "/models/test.gguf" {
		t.Errorf("expected model path /models/test.gguf, got %q", c.ModelPath)
	}
	want := []string{"http://a.example", "http://b.example"}
	if !reflect.DeepEqual(c.AllowedOrigins, want) {
		t.Errorf("expected origins %v, got %v", want, c.AllowedOrigins)
	}
}

// TestApplyEnv_UnsetLeavesDefaults is the regression guard for the original
// bug: an unset variable must not clobber a value that was set explicitly
// (for example by a flag).
func TestApplyEnv_UnsetLeavesDefaults(t *testing.T) {
	// t.Setenv with "" still sets the variable, so clear them explicitly.
	for _, k := range []string{EnvPort, EnvMetricsPort, EnvHost, EnvAPIKey, EnvAllowedOrigins, EnvModelPath} {
		t.Setenv(k, "")
		if err := unsetenv(k); err != nil {
			t.Fatalf("failed to unset %s: %v", k, err)
		}
	}

	c := DefaultConfig()
	c.APIKey = "set-by-flag"
	c.Port = 1234
	c.ApplyEnv()

	if c.APIKey != "set-by-flag" {
		t.Errorf("ApplyEnv overwrote an explicitly set API key: %q", c.APIKey)
	}
	if c.Port != 1234 {
		t.Errorf("ApplyEnv overwrote an explicitly set port: %d", c.Port)
	}
	if c.Host != "0.0.0.0" {
		t.Errorf("expected default host, got %q", c.Host)
	}
}

func TestApplyEnv_IgnoresInvalidPort(t *testing.T) {
	t.Setenv(EnvPort, "not-a-number")

	c := DefaultConfig()
	c.ApplyEnv()

	if c.Port != 8080 {
		t.Errorf("expected the default port to survive an unparseable value, got %d", c.Port)
	}
}

func TestSplitOrigins(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"", []string{}},
		{"   ", []string{}},
		{"a", []string{"a"}},
		{"a,b", []string{"a", "b"}},
		{" a , b ,", []string{"a", "b"}},
	}
	for _, tc := range cases {
		got := SplitOrigins(tc.in)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("SplitOrigins(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
}
