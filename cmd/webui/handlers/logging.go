//go:build webui

package handlers

import (
	"io"
	"log/slog"
	"net/http"
	"time"
)

type Logger struct {
	logger *slog.Logger
}

func NewLogger() *Logger {
	logger := slog.New(slog.NewJSONHandler(io.Discard, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	}))
	return &Logger{logger: logger}
}

func (l *Logger) With(handler slog.Handler) *Logger {
	return &Logger{logger: slog.New(handler)}
}

type RequestLog struct {
	RequestID     string    `json:"request_id"`
	Timestamp     time.Time `json:"timestamp"`
	Method        string    `json:"method"`
	Path          string    `json:"path"`
	Query         string    `json:"query,omitempty"`
	StatusCode    int       `json:"status_code"`
	Duration      float64   `json:"duration_ms"`
	ContentLength int       `json:"content_length"`
	UserAgent     string    `json:"user_agent,omitempty"`
	ClientIP      string    `json:"client_ip"`
	Error         string    `json:"error,omitempty"`
}

type LoggingMiddleware struct {
	logger    *slog.Logger
	skipPaths map[string]bool
}

func NewLoggingMiddleware() *LoggingMiddleware {
	return &LoggingMiddleware{
		logger: slog.New(slog.NewJSONHandler(io.Discard, nil)),
		skipPaths: map[string]bool{
			"/health":      true,
			"/healthz":     true,
			"/readyz":      true,
			"/metrics":     true,
			"/favicon.ico": true,
		},
	}
}

func (m *LoggingMiddleware) Middleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if m.skipPaths[r.URL.Path] {
			next.ServeHTTP(w, r)
			return
		}

		requestID := generateRequestID()
		start := time.Now()

		w.Header().Set("X-Request-ID", requestID)

		next.ServeHTTP(w, r)

		duration := time.Since(start)

		logEntry := RequestLog{
			RequestID:     requestID,
			Timestamp:     start,
			Method:        r.Method,
			Path:          r.URL.Path,
			Query:         r.URL.RawQuery,
			ContentLength: int(r.ContentLength),
			UserAgent:     r.UserAgent(),
			ClientIP:      r.RemoteAddr,
			Duration:      duration.Seconds() * 1000,
		}

		m.logger.Info("HTTP Request", slog.Any("request", logEntry))
	}
}

func generateRequestID() string {
	return time.Now().Format("20060102150405")[:14]
}

func (m *LoggingMiddleware) LogError(requestID string, err error, context map[string]interface{}) {
	m.logger.Error("Error occurred",
		slog.String("request_id", requestID),
		slog.String("error", err.Error()),
		slog.Any("context", context),
	)
}

type StructuredLogger struct {
	logger *slog.Logger
}

func NewStructuredLogger() *StructuredLogger {
	return &StructuredLogger{
		logger: slog.New(slog.NewJSONHandler(io.Discard, &slog.HandlerOptions{
			Level: slog.LevelInfo,
		})),
	}
}

func (l *StructuredLogger) WithComponent(component string) *StructuredLogger {
	return &StructuredLogger{
		logger: l.logger.With(slog.String("component", component)),
	}
}

func (l *StructuredLogger) Debug(msg string, args ...any) {
	l.logger.Debug(msg, args...)
}

func (l *StructuredLogger) Info(msg string, args ...any) {
	l.logger.Info(msg, args...)
}

func (l *StructuredLogger) Warn(msg string, args ...any) {
	l.logger.Warn(msg, args...)
}

func (l *StructuredLogger) Error(msg string, args ...any) {
	l.logger.Error(msg, args...)
}
