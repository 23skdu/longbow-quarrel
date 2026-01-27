package logger

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/rs/zerolog"
)

// Log is the global logger instance wrapper
var Log *Logger

type Logger struct {
	z zerolog.Logger
}

func init() {
	// Default setup
	output := zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339}
	z := zerolog.New(output).With().Timestamp().Logger()
	Log = &Logger{z: z}
}

// Setup configures the global logger
func Setup(level string, format string) {
	var logLevel zerolog.Level
	switch strings.ToUpper(level) {
	case "DEBUG":
		logLevel = zerolog.DebugLevel
	case "WARN":
		logLevel = zerolog.WarnLevel
	case "ERROR":
		logLevel = zerolog.ErrorLevel
	default:
		logLevel = zerolog.InfoLevel
	}

	zerolog.SetGlobalLevel(logLevel)

	var z zerolog.Logger
	if strings.ToLower(format) == "json" {
		z = zerolog.New(os.Stderr).With().Timestamp().Logger()
	} else {
		output := zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339}
		z = zerolog.New(output).With().Timestamp().Logger()
	}

	Log = &Logger{z: z}
}

// Info logs at Info level with variadic key-value pairs
func (l *Logger) Info(msg string, args ...interface{}) {
	e := l.z.Info()
	addFields(e, args...)
	e.Msg(msg)
}

// Debug logs at Debug level with variadic key-value pairs
func (l *Logger) Debug(msg string, args ...interface{}) {
	e := l.z.Debug()
	addFields(e, args...)
	e.Msg(msg)
}

// Warn logs at Warn level with variadic key-value pairs
func (l *Logger) Warn(msg string, args ...interface{}) {
	e := l.z.Warn()
	addFields(e, args...)
	e.Msg(msg)
}

// Error logs at Error level with variadic key-value pairs
func (l *Logger) Error(msg string, args ...interface{}) {
	e := l.z.Error()
	addFields(e, args...)
	e.Msg(msg)
}

// addFields adds variadic key-value pairs to the event
func addFields(e *zerolog.Event, args ...interface{}) {
	for i := 0; i < len(args); i += 2 {
		if i+1 < len(args) {
			key, ok := args[i].(string)
			if !ok {
				key = fmt.Sprintf("%v", args[i])
			}
			e.Interface(key, args[i+1])
		}
	}
}
