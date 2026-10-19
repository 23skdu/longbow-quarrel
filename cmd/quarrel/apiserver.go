package main

import (
	"fmt"
	"net/http"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/api"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

// apiServer bundles the REST API server with the address it listens on, so
// callers (and tests) can shut it down deterministically.
type apiServer struct {
	addr string
	srv  *http.Server
}

// startAPIServer starts the OpenAI-compatible REST API on addr.
//
// The API used to be registered on http.DefaultServeMux by
// api.InitServer and was called from exactly one place: a test. No shipped
// binary ever started it, so the endpoints documented in README.md and
// docs/api.md did not exist in any released artifact.
//
// addr may be empty, in which case the API is disabled and (nil, nil) is
// returned.
//
// specEngine is the optional speculative-decoding engine. When it is given,
// the requests that set "speculative" use it.
func startAPIServer(addr string, memFn func() int64, e engine.Engine, specEngine engine.Engine, tok *tokenizer.Tokenizer) (*apiServer, error) {
	if addr == "" {
		return nil, nil
	}

	mux := api.InitServerWithSpeculative(0, memFn, e, specEngine, tok)
	srv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Log.Error("REST API server exited", "error", err)
		}
	}()

	return &apiServer{addr: addr, srv: srv}, nil
}

func (a *apiServer) Close() {
	if a == nil || a.srv == nil {
		return
	}
	_ = a.srv.Close()
}

// describe returns a human-readable summary of the API endpoints.
func (a *apiServer) describe() string {
	if a == nil {
		return ""
	}
	return fmt.Sprintf("REST API: http://localhost%s/v1/chat/completions\nHealth:   http://localhost%s/healthz", a.addr, a.addr)
}
