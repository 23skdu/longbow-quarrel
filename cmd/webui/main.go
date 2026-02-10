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
	port        = flag.Int("port", 8080, "HTTP server port")
	metricsPort = flag.Int("metrics-port", 9090, "Prometheus metrics port")
	host        = flag.String("host", "0.0.0.0", "Host to bind to")
)

func main() {
	flag.Parse()

	cfg := config.Config{
		Port:        *port,
		MetricsPort: *metricsPort,
		Host:        *host,
	}

	log.Printf("Starting Longbow-Quarrel WebUI on %s:%d", cfg.Host, *port)
	log.Printf("Metrics available at http://%s:%d/metrics", cfg.Host, *metricsPort)

	if err := templates.InitTemplates(); err != nil {
		log.Fatalf("Failed to initialize templates: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mux := http.NewServeMux()

	mux.Handle("/", handlers.IndexHandler())
	mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))
	mux.Handle("/ws", handlers.WebSocketHandler(cfg))
	mux.Handle("/api/models", handlers.ModelsHandler(cfg))
	mux.Handle("/api/generate", handlers.GenerateHandler(cfg))
	mux.Handle("/api/stream", handlers.StreamHandler(cfg))
	mux.Handle("/health", handlers.HealthHandler())
	mux.Handle("/metrics", promhttp.Handler())

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
