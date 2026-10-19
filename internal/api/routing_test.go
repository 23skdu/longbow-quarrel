package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestInitServer_RoutesAllEndpoints asserts that every endpoint documented in
// README.md and docs/api.md is actually registered on the mux returned by
// InitServer.
//
// InitServer used to register on http.DefaultServeMux and was called from
// exactly one place: a test. No shipped binary ever started it, so the
// documented OpenAI-compatible surface did not exist. This test fails if an
// endpoint is dropped or renamed.
func TestInitServer_RoutesAllEndpoints(t *testing.T) {
	mux := InitServer(1024, func() int64 { return 0 }, &mockEngine{}, &mockTokenizer{})
	if mux == nil {
		t.Fatal("InitServer returned nil mux")
	}

	cases := []struct {
		method string
		path   string
		body   string
	}{
		{http.MethodGet, "/healthz", ""},
		{http.MethodGet, "/readyz", ""},
		{http.MethodPost, "/v1/completions", `{"prompt":"hello","max_tokens":2}`},
		{http.MethodPost, "/v1/chat/completions", `{"messages":[{"role":"user","content":"hello"}]}`},
		{http.MethodGet, "/v1/adapters/list", ""},
		{http.MethodPost, "/v1/adapters/load", `{"path":"/tmp/a.gguf","id":"a"}`},
	}

	for _, tc := range cases {
		t.Run(tc.method+" "+tc.path, func(t *testing.T) {
			var req *http.Request
			if tc.body == "" {
				req = httptest.NewRequest(tc.method, tc.path, nil)
			} else {
				req = httptest.NewRequest(tc.method, tc.path, bytes.NewBufferString(tc.body))
				req.Header.Set("Content-Type", "application/json")
			}
			rr := httptest.NewRecorder()

			mux.ServeHTTP(rr, req)

			// 404 means the route was never registered.
			if rr.Code == http.StatusNotFound {
				t.Fatalf("endpoint %s %s is not registered on the mux", tc.method, tc.path)
			}
			// A wrong method must still be handled (405), not 404.
			if rr.Code == http.StatusMethodNotAllowed {
				t.Fatalf("endpoint %s %s rejected its own method", tc.method, tc.path)
			}
		})
	}
}

// TestInitServer_HealthzReportsUsage exercises the health endpoint through
// the mux, which is how Kubernetes probes reach it.
func TestInitServer_HealthzReportsUsage(t *testing.T) {
	mux := InitServer(1000, func() int64 { return 250 }, &mockEngine{}, &mockTokenizer{})

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 from /healthz, got %d", rr.Code)
	}

	var resp HealthResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode health response: %v", err)
	}
	if resp.LoadPercent != 25 {
		t.Errorf("expected 25%% load, got %d%%", resp.LoadPercent)
	}
	if resp.MemoryTotal != 1000 || resp.MemoryUsed != 250 {
		t.Errorf("unexpected memory report: used=%d total=%d", resp.MemoryUsed, resp.MemoryTotal)
	}
}

// TestInitServer_IsRepeatable ensures InitServer can be called more than
// once in a process. Registering on http.DefaultServeMux made the second
// call panic with "multiple registrations for /healthz", which is part of
// why the API was never wired into a binary.
func TestInitServer_IsRepeatable(t *testing.T) {
	for i := 0; i < 3; i++ {
		mux := InitServer(1024, func() int64 { return 0 }, &mockEngine{}, &mockTokenizer{})
		if mux == nil {
			t.Fatalf("InitServer call %d returned nil mux", i)
		}
	}
}

// TestServer_ServesOverHTTP starts the mux on a real listener and makes a
// real HTTP request, proving the API is reachable the way a client or a
// Kubernetes probe would reach it.
func TestServer_ServesOverHTTP(t *testing.T) {
	mux := InitServer(1024, func() int64 { return 0 }, &mockEngine{}, &mockTokenizer{})

	srv := httptest.NewServer(mux)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz failed: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 from /healthz, got %d", resp.StatusCode)
	}

	var health HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		t.Fatalf("failed to decode health response: %v", err)
	}
	if health.Status != "ok" {
		t.Errorf("expected status 'ok', got %q", health.Status)
	}
}
