package logger

import (
	"bytes"
	"strings"
	"testing"

	"github.com/rs/zerolog"
)

// newCapturingLogger returns a Logger writing JSON lines into an in-memory
// buffer so the assertions below can inspect what was actually logged. It also
// raises the global level to debug so nothing under test gets filtered out.
func newCapturingLogger(t *testing.T) (*Logger, *bytes.Buffer) {
	t.Helper()
	Setup("debug", "console")
	var buf bytes.Buffer
	return &Logger{z: zerolog.New(&buf).With().Timestamp().Logger()}, &buf
}

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
	l, buf := newCapturingLogger(t)

	l.Info("test info message", "key", "value")
	l.Debug("test debug message", "key", "value")
	l.Warn("test warn message", "key", "value")
	l.Error("test error message", "key", "value")

	out := buf.String()
	for _, msg := range []string{
		"test info message", "test debug message", "test warn message", "test error message",
	} {
		if !strings.Contains(out, msg) {
			t.Errorf("log output does not contain %q:\n%s", msg, out)
		}
	}
	for _, level := range []string{
		`"level":"info"`, `"level":"debug"`, `"level":"warn"`, `"level":"error"`,
	} {
		if !strings.Contains(out, level) {
			t.Errorf("log output does not contain level %s:\n%s", level, out)
		}
	}
}
func TestLoggerWithMultipleFields(t *testing.T) {
	l, buf := newCapturingLogger(t)

	l.Info(
		"multi-field test",
		"string_field", "value",
		"int_field", 42,
		"float_field", 3.14,
		"bool_field", true,
	)

	out := buf.String()
	for _, want := range []string{
		`"string_field":"value"`,
		`"int_field":42`,
		`"float_field":3.14`,
		`"bool_field":true`,
		"multi-field test",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("log output does not contain %s:\n%s", want, out)
		}
	}
}
func TestLoggerWithNoFields(t *testing.T) {
	l, buf := newCapturingLogger(t)

	l.Info("no fields message")
	l.Debug("debug no fields")
	l.Warn("warn no fields")
	l.Error("error no fields")

	out := buf.String()
	for _, msg := range []string{"no fields message", "debug no fields", "warn no fields", "error no fields"} {
		if !strings.Contains(out, msg) {
			t.Errorf("log output does not contain %q:\n%s", msg, out)
		}
	}
	if lines := strings.Count(strings.TrimSpace(out), "\n"); lines != 3 {
		t.Errorf("wrote %d log lines, want 4:\n%s", lines+1, out)
	}
}
func TestLoggerWithOddArgs(t *testing.T) {
	l, buf := newCapturingLogger(t)

	// The trailing key has no value, so addFields must drop it rather than
	// pair it with whatever comes next.
	l.Info("odd args", "key1", "value1", "orphan_key")

	out := buf.String()
	if !strings.Contains(out, `"key1":"value1"`) {
		t.Errorf("log output is missing the complete pair:\n%s", out)
	}
	if strings.Contains(out, "orphan_key") {
		t.Errorf("log output kept a key that has no value:\n%s", out)
	}
}
func TestLoggerWithEmptyArgs(t *testing.T) {
	l, buf := newCapturingLogger(t)

	l.Info("empty args")

	if out := buf.String(); !strings.Contains(out, "empty args") {
		t.Errorf("log output does not contain the message:\n%s", out)
	}
}
func TestLoggerLevelFiltering(t *testing.T) {
	Setup("error", "console")
	if got := zerolog.GlobalLevel(); got != zerolog.ErrorLevel {
		t.Fatalf("global level = %v, want %v", got, zerolog.ErrorLevel)
	}

	var buf bytes.Buffer
	l := &Logger{z: zerolog.New(&buf).With().Timestamp().Logger()}
	l.Error("error message should appear")
	l.Debug("debug message should be filtered")
	l.Info("info message should be filtered")
	l.Warn("warn message should be filtered")

	out := buf.String()
	if !strings.Contains(out, "error message should appear") {
		t.Errorf("error was filtered out:\n%s", out)
	}
	for _, filtered := range []string{"debug message should be filtered", "info message should be filtered", "warn message should be filtered"} {
		if strings.Contains(out, filtered) {
			t.Errorf("%q should have been filtered:\n%s", filtered, out)
		}
	}
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
	want := map[string]zerolog.Level{
		"DEBUG": zerolog.DebugLevel,
		"Debug": zerolog.DebugLevel,
		"debug": zerolog.DebugLevel,
		"Info":  zerolog.InfoLevel,
		"INFO":  zerolog.InfoLevel,
		"info":  zerolog.InfoLevel,
	}

	for level, expect := range want {
		Setup(level, "console")
		if got := zerolog.GlobalLevel(); got != expect {
			t.Errorf("Setup(%q) set level %v, want %v", level, got, expect)
		}
	}
}
func TestAddFieldsWithNonStringKey(t *testing.T) {
	l, buf := newCapturingLogger(t)

	l.Info("test non-string key", 123, "value")

	out := buf.String()
	if !strings.Contains(out, `"123":"value"`) {
		t.Errorf("non-string key was not stringified into a field:\n%s", out)
	}
	if !strings.Contains(out, "test non-string key") {
		t.Errorf("log output does not contain the message:\n%s", out)
	}
}
func TestAddFieldsWithNilValue(t *testing.T) {
	l, buf := newCapturingLogger(t)

	l.Info("test nil value", "key", nil)

	out := buf.String()
	if !strings.Contains(out, `"key":null`) {
		t.Errorf("nil value was not logged as null:\n%s", out)
	}
}
func TestLoggerStructFields(t *testing.T) {
	// The wrapper delegates everything to its zerolog.Logger, so a logger built
	// around a buffer must be the one that ends up writing.
	l, buf := newCapturingLogger(t)

	l.Info("struct field probe", "k", "v")

	out := buf.String()
	if !strings.Contains(out, "struct field probe") {
		t.Errorf("wrapped logger did not write the message:\n%s", out)
	}
	if !strings.Contains(out, `"k":"v"`) {
		t.Errorf("wrapped logger did not write the field:\n%s", out)
	}
}
