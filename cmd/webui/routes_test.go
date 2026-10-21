//go:build webui

package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/23skdu/longbow-quarrel/cmd/webui/config"
	"github.com/23skdu/longbow-quarrel/cmd/webui/templates"
)

// TestTheRoutesAreReachable checks the routes through the real router.
//
// The end-to-end suite called the handlers directly, so it never noticed
// that the API was mounted under /api/ without the prefix being
// stripped: http.ServeMux matches the whole path, so every API endpoint
// answered 404 and the browser could not use the server at all.
func TestTheRoutesAreReachable(t *testing.T) {
	// The index page is rendered from a template that has to be
	// initialized first.
	if err := templates.InitTemplates(); err != nil {
		t.Fatalf("failed to initialize the templates: %v", err)
	}

	mux := newMux(config.DefaultConfig())

	tests := []struct {
		method string
		path   string
		body   string
		want   int
	}{
		// The probes and the version.
		{method: http.MethodGet, path: "/health", want: http.StatusOK},
		{method: http.MethodGet, path: "/healthz", want: http.StatusOK},
		{method: http.MethodGet, path: "/readyz", want: http.StatusOK},
		{method: http.MethodGet, path: "/version", want: http.StatusOK},

		// The API. A model is not loaded, so the requests that need one
		// fail, but they must not be missing.
		{method: http.MethodGet, path: "/api/models", want: http.StatusOK},
		{method: http.MethodGet, path: "/api/adapters", want: http.StatusOK},
		{method: http.MethodGet, path: "/api/v1/models", want: http.StatusOK},
		{method: http.MethodPost, path: "/api/generate", body: `{"prompt":"hi"}`,
			want: http.StatusInternalServerError},
		// The stream endpoint answers with the SSE headers and reports the
		// failure inside the stream, so its status is 200.
		{method: http.MethodPost, path: "/api/stream", body: `{"prompt":"hi"}`,
			want: http.StatusOK},

		// The index page.
		{method: http.MethodGet, path: "/", want: http.StatusOK},
	}

	for _, tt := range tests {
		t.Run(tt.method+" "+tt.path, func(t *testing.T) {
			var body *strings.Reader
			if tt.body != "" {
				body = strings.NewReader(tt.body)
			} else {
				body = strings.NewReader("")
			}
			req := httptest.NewRequest(tt.method, tt.path, body)
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			mux.ServeHTTP(w, req)

			if w.Code != tt.want {
				t.Errorf("the status is %d, want %d (body: %s)", w.Code, tt.want, w.Body.String())
			}
		})
	}
}

// TestTheAPIRejectsUnknownFields checks that a typo in a parameter name
// is reported instead of being ignored.
func TestTheAPIRejectsUnknownFields(t *testing.T) {
	mux := newMux(config.DefaultConfig())

	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"promt":"hi"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("the status is %d, want %d", w.Code, http.StatusBadRequest)
	}
	if ct := w.Header().Get("Content-Type"); !strings.Contains(ct, "application/json") {
		t.Errorf("the content type is %q, want application/json", ct)
	}
}
