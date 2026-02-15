package logger

import (
	"testing"

	"github.com/rs/zerolog"
)

func TestSetup(t *testing.T) {
	tests := []struct {
		name   string
		level  string
		format string
	}{
		{"debug level", "debug", "console"},
		{"info level", "info", "console"},
		{"warn level", "warn", "console"},
		{"error level", "error", "console"},
		{"json format", "info", "json"},
		{"lowercase level", "debug", "console"},
		{"uppercase level", "DEBUG", "console"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Setup should not panic
			Setup(tt.level, tt.format)
			if Log == nil {
				t.Error("expected Log to be initialized")
			}
		})
	}
}

func TestLoggerMethodsExist(t *testing.T) {
	// Test that all logger methods exist and can be called
	if Log == nil {
		Setup("info", "console")
	}

	// These should not panic
	Log.Info("test info message", "key", "value")
	Log.Debug("test debug message", "key", "value")
	Log.Warn("test warn message", "key", "value")
	Log.Error("test error message", "key", "value")
}

func TestLoggerWithMultipleFields(t *testing.T) {
	if Log == nil {
		Setup("debug", "console")
	}

	// Test with multiple key-value pairs
	Log.Info(
		"multi-field test",
		"string_field", "value",
		"int_field", 42,
		"float_field", 3.14,
		"bool_field", true,
	)
}

func TestLoggerWithNoFields(t *testing.T) {
	if Log == nil {
		Setup("info", "console")
	}

	// Test with no additional fields
	Log.Info("no fields message")
	Log.Debug("debug no fields")
	Log.Warn("warn no fields")
	Log.Error("error no fields")
}

func TestLoggerWithOddArgs(t *testing.T) {
	if Log == nil {
		Setup("info", "console")
	}

	// Test with odd number of args (last key without value)
	Log.Info("odd args", "key1", "value1", "orphan_key")
}

func TestLoggerWithEmptyArgs(t *testing.T) {
	if Log == nil {
		Setup("info", "console")
	}

	// Test with empty args slice
	Log.Info("empty args")
}

func TestLoggerLevelFiltering(t *testing.T) {
	// Setup with error level - debug and info should be filtered
	Setup("error", "console")

	// These should not panic even though they may be filtered
	Log.Error("error message should appear")
	Log.Debug("debug message should be filtered")
	Log.Info("info message should be filtered")
	Log.Warn("warn message should be filtered")
}

func TestLoggerLevelConstants(t *testing.T) {
	tests := []struct {
		level  string
		expect zerolog.Level
	}{
		{"debug", zerolog.DebugLevel},
		{"info", zerolog.InfoLevel},
		{"warn", zerolog.WarnLevel},
		{"error", zerolog.ErrorLevel},
		{"unknown", zerolog.InfoLevel}, // default case
	}

	for _, tt := range tests {
		t.Run(tt.level, func(t *testing.T) {
			Setup(tt.level, "console")
			got := zerolog.GlobalLevel()
			if got != tt.expect {
				t.Errorf("level %s: expected %v, got %v", tt.level, tt.expect, got)
			}
		})
	}
}

func TestLoggerFormatJSON(t *testing.T) {
	// Setup with JSON format
	Setup("info", "json")
	if Log == nil {
		t.Error("expected Log to be initialized")
	}
}

func TestLoggerFormatConsole(t *testing.T) {
	// Setup with console format (default)
	Setup("info", "console")
	if Log == nil {
		t.Error("expected Log to be initialized")
	}
}

func TestLoggerCaseInsensitiveLevel(t *testing.T) {
	levels := []string{"DEBUG", "Debug", "debug", "Info", "INFO", "info"}

	for _, level := range levels {
		Setup(level, "console")
		// Just verify no panic
		Log.Info("test")
	}
}

func TestAddFieldsWithNonStringKey(t *testing.T) {
	if Log == nil {
		Setup("info", "console")
	}

	// Test with non-string key (should be converted to string)
	Log.Info("test non-string key", 123, "value")
}

func TestAddFieldsWithNilValue(t *testing.T) {
	if Log == nil {
		Setup("info", "console")
	}

	// Test with nil value
	Log.Info("test nil value", "key", nil)
}

func TestLoggerStructFields(t *testing.T) {
	// Verify Logger struct can be instantiated
	l := &Logger{}
	_ = l // Use the variable to avoid unused warning
}
