package api

import (
	"bytes"
	"errors"
	"net/http"
	"testing"
)

type faultyResponseWriter struct {
	header http.Header
	status int
}

func (f *faultyResponseWriter) Header() http.Header {
	if f.header == nil {
		f.header = make(http.Header)
	}
	return f.header
}

func (f *faultyResponseWriter) Write(b []byte) (int, error) {
	return 0, errors.New("simulated write failure")
}

func (f *faultyResponseWriter) WriteHeader(statusCode int) {
	f.status = statusCode
}

func TestServer_FaultInjection(t *testing.T) {
	s := &Server{
		MaxMemory:  1000,
		UsedMemory: func() int64 { return 10 },
	}

	t.Run("JSONEncodingFailure", func(t *testing.T) {
		w := &faultyResponseWriter{}
		r, _ := http.NewRequest("GET", "/healthz", nil)

		s.HealthzEndpoint(w, r)
		_ = w.status // Ensure used
	})
	
	t.Run("CompletionsMalformedJSON", func(t *testing.T) {
		w := &faultyResponseWriter{}
		r, _ := http.NewRequest("POST", "/v1/completions", bytes.NewBufferString("{invalid json"))
		s.CompletionsHandler(w, r)
		if w.status == 0 {
			// Just a use of w.status to satisfy compiler
		}
	})
}
