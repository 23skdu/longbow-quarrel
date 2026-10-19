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

	"net/http/pprof"

	"github.com/23skdu/longbow-quarrel/cmd/webui/config"
	"github.com/23skdu/longbow-quarrel/cmd/webui/handlers"
	"github.com/23skdu/longbow-quarrel/cmd/webui/templates"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	port           = flag.Int("port", 0, "HTTP server port (default 8080, or $WEBUI_PORT)")
	metricsPort    = flag.Int("metrics-port", 0, "Prometheus metrics port (default 9090, or $WEBUI_METRICS_PORT)")
	host           = flag.String("host", "", "Host to bind to (default 0.0.0.0, or $WEBUI_HOST)")
	apiKey         = flag.String("api-key", "", "API key for authentication (or $WEBUI_API_KEY)")
	allowedOrigins = flag.String("allowed-origins", "", "Comma-separated list of allowed CORS origins (or $WEBUI_ALLOWED_ORIGINS)")
)

func main() {
	flag.Parse()

	// Flags win over the environment; the environment wins over the
	// defaults. Dockerfile.webui and docker-compose.webui.yml set
	// WEBUI_* variables, but nothing read them, so WEBUI_API_KEY silently
	// produced an unauthenticated server.
	cfg := config.DefaultConfig()
	cfg.ApplyEnv()

	// A flag that was explicitly provided overrides the environment.
	flag.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "port":
			cfg.Port = *port
		case "metrics-port":
			cfg.MetricsPort = *metricsPort
		case "host":
			cfg.Host = *host
		case "api-key":
			cfg.APIKey = *apiKey
		case "allowed-origins":
			cfg.AllowedOrigins = config.SplitOrigins(*allowedOrigins)
		}
	})

	log.Printf("Starting Longbow-Quarrel WebUI v%s on %s:%d", handlers.Version, cfg.Host, cfg.Port)
	log.Printf("Metrics available at http://%s:%d/metrics", cfg.Host, cfg.MetricsPort)
	if cfg.APIKey == "" {
		log.Printf("WARNING: no API key configured (set -api-key or $%s); the API is unauthenticated", config.EnvAPIKey)
	}

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
	mux.Handle("/metrics", loggingMiddleware.Middleware(http.HandlerFunc(promhttp.Handler().ServeHTTP)))

	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
	mux.Handle("/debug/pprof/allocs", pprof.Handler("allocs"))
	mux.Handle("/debug/pprof/heap", pprof.Handler("heap"))
	mux.Handle("/debug/pprof/goroutine", pprof.Handler("goroutine"))
	mux.Handle("/debug/pprof/threadcreate", pprof.Handler("threadcreate"))
	mux.Handle("/debug/pprof/block", pprof.Handler("block"))
	mux.Handle("/debug/pprof/mutex", pprof.Handler("mutex"))

	apiMux := http.NewServeMux()
	apiMux.Handle("/models", authMiddleware.Authenticate(handlers.ModelsHandler(cfg)))
	apiMux.Handle("/generate", authMiddleware.Authenticate(handlers.GenerateHandler(cfg)))
	apiMux.Handle("/stream", authMiddleware.Authenticate(handlers.StreamHandler(cfg)))
	apiMux.Handle("/hotswap", authMiddleware.Authenticate(handlers.HotSwapHandler(cfg)))
	apiMux.Handle("/v1/chat/completions", authMiddleware.Authenticate(handlers.OpenAIChatCompletionsHandler(cfg)))
	apiMux.Handle("/v1/completions", authMiddleware.Authenticate(handlers.OpenAICompletionsHandler(cfg)))
	apiMux.Handle("/v1/models", authMiddleware.Authenticate(handlers.OpenAIModelsHandler(cfg)))
	mux.Handle("/api/", loggingMiddleware.Middleware(corsMiddleware.Middleware(apiMux.ServeHTTP)))

	mux.Handle("/", handlers.IndexHandler())
	mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))
	mux.Handle("/ws", handlers.WebSocketHandler(cfg))

	server := &http.Server{
		Addr:    fmt.Sprintf("%s:%d", cfg.Host, cfg.Port),
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
