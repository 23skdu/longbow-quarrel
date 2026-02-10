//go:build webui

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/23skdu/longbow-quarrel/cmd/webui/config"
	"github.com/23skdu/longbow-quarrel/cmd/webui/handlers"
	"github.com/23skdu/longbow-quarrel/cmd/webui/templates"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	port           = flag.Int("port", 8080, "HTTP server port")
	metricsPort    = flag.Int("metrics-port", 9090, "Prometheus metrics port")
	host           = flag.String("host", "0.0.0.0", "Host to bind to")
	apiKey         = flag.String("api-key", "", "API key for authentication")
	allowedOrigins = flag.String("allowed-origins", "", "Comma-separated list of allowed CORS origins")
)

func main() {
	flag.Parse()

	cfg := config.Config{
		Port:           *port,
		MetricsPort:    *metricsPort,
		Host:           *host,
		APIKey:         *apiKey,
		AllowedOrigins: parseOrigins(*allowedOrigins),
	}

	log.Printf("Starting Longbow-Quarrel WebUI v%s on %s:%d", handlers.Version, cfg.Host, *port)
	log.Printf("Metrics available at http://%s:%d/metrics", cfg.Host, *metricsPort)

	if err := templates.InitTemplates(); err != nil {
		log.Fatalf("Failed to initialize templates: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mux := http.NewServeMux()

	corsMiddleware := handlers.NewCORSMiddleware(cfg.AllowedOrigins)
	authMiddleware := handlers.NewAuthMiddleware(cfg.APIKey)
	loggingMiddleware := handlers.NewLoggingMiddleware()

	mux.Handle("/health", handlers.HealthHandler())
	mux.Handle("/healthz", handlers.HealthzHandler())
	mux.Handle("/readyz", handlers.ReadyzHandler())
	mux.Handle("/version", handlers.VersionHandler())
	mux.Handle("/metrics", loggingMiddleware.Middleware(promhttp.Handler()))

	apiMux := http.NewServeMux()
	apiMux.Handle("/models", authMiddleware.Authenticate(handlers.ModelsHandler(cfg)))
	apiMux.Handle("/generate", authMiddleware.Authenticate(handlers.GenerateHandler(cfg)))
	apiMux.Handle("/stream", authMiddleware.Authenticate(handlers.StreamHandler(cfg)))
	mux.Handle("/api/", loggingMiddleware.Middleware(corsMiddleware.Middleware(apiMux.ServeHTTP)))

	mux.Handle("/", handlers.IndexHandler())
	mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))
	mux.Handle("/ws", handlers.WebSocketHandler(cfg))

	server := &http.Server{
		Addr:    fmt.Sprintf("%s:%d", cfg.Host, *port),
		Handler: mux,
	}

	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
		<-sigChan
		log.Println("Shutting down server...")
		cancel()
		server.Shutdown(context.Background())
	}()

	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}

	<-ctx.Done()
	log.Println("Server stopped")
}

func parseOrigins(origins string) []string {
	if origins == "" {
		return []string{}
	}
	var result []string
	for _, origin := range splitString(origins, ',') {
		if trimmed := trimSpace(origin); trimmed != "" {
			result = append(result, trimmed)
		}
	}
	return result
}

func splitString(s string, sep byte) []string {
	var result []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == sep {
			result = append(result, s[start:i])
			start = i + 1
		}
	}
	result = append(result, s[start:])
	return result
}

func trimSpace(s string) string {
	start := 0
	end := len(s)
	for start < end && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r') {
		start++
	}
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n' || s[end-1] == '\r') {
		end--
	}
	return s[start:end]
}
