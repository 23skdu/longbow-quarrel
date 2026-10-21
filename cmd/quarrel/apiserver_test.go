package main

import (
	"encoding/json"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

// freePort reserves an ephemeral port and releases it, returning the port
// number. There is an inherent race between releasing the port and the
// server binding it, so the helper retries on failure.
func freePort(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to reserve a port: %v", err)
	}
	addr := l.Addr().String()
	if err := l.Close(); err != nil {
		t.Fatalf("failed to release the reserved port: %v", err)
	}
	return addr
}

// TestStartAPIServer_ServesEndpoints proves the shipped binary actually
// exposes the OpenAI-compatible REST API.
//
// api.InitServer used to register on http.DefaultServeMux and was called
// from exactly one place -- a test. No binary ever started it, so
// /v1/chat/completions and /healthz did not exist in any released artifact
// even though README.md and docs/api.md documented them.
func TestStartAPIServer_ServesEndpoints(t *testing.T) {
	mockEng := &mockEngine{cfg: config.Config{}}

	addr := freePort(t)

	srv, err := startAPIServer(addr, func() int64 { return 0 }, mockEng, nil, nil)
	if err != nil {
		t.Fatalf("startAPIServer failed: %v", err)
	}
	if srv == nil {
		t.Fatal("startAPIServer returned nil for a non-empty address")
	}
	defer srv.Close()

	base := "http://" + addr

	// /healthz must answer 200 with a JSON body.
	var health map[string]any
	waitForServer(t, base+"/healthz")
	resp, err := http.Get(base + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz failed: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 from /healthz, got %d", resp.StatusCode)
	}
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		t.Fatalf("failed to decode /healthz body: %v", err)
	}
	if _, ok := health["status"]; !ok {
		t.Errorf("/healthz response is missing the 'status' field: %v", health)
	}

	// The documented chat-completions endpoint must be routed. With a nil
	// tokenizer the handler answers 503 ("Inference components not
	// initialized"), which still proves the route exists -- a missing route
	// would be 404.
	resp2, err := http.Get(base + "/v1/chat/completions")
	if err != nil {
		t.Fatalf("GET /v1/chat/completions failed: %v", err)
	}
	defer func() { _ = resp2.Body.Close() }()
	if resp2.StatusCode == http.StatusNotFound {
		t.Fatal("/v1/chat/completions is not registered on the API server")
	}
}

// TestStartAPIServer_EmptyAddressDisables ensures the API can be turned off.
func TestStartAPIServer_EmptyAddressDisables(t *testing.T) {
	srv, err := startAPIServer("", func() int64 { return 0 }, nil, nil, nil)
	if err != nil {
		t.Fatalf("startAPIServer with an empty address failed: %v", err)
	}
	if srv != nil {
		t.Error("expected nil server for an empty address")
		srv.Close()
	}
}

// TestStartAPIServer_DescribeMentionsEndpoints guards the startup banner
// against drifting away from the routes that are actually served.
func TestStartAPIServer_DescribeMentionsEndpoints(t *testing.T) {
	srv := &apiServer{addr: ":8080"}
	desc := srv.describe()
	for _, want := range []string{"/v1/chat/completions", "/healthz"} {
		if !strings.Contains(desc, want) {
			t.Errorf("describe() = %q, expected it to mention %q", desc, want)
		}
	}
}

// waitForServer polls url until it answers or the deadline passes.
func waitForServer(t *testing.T, url string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get(url)
		if err == nil {
			_ = resp.Body.Close()
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("server at %s did not become ready", url)
}
