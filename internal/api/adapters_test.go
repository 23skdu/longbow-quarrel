package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

type mockAdapterEngine struct {
	mockEngine
	loadErr error
}

func (m *mockAdapterEngine) LoadAdapter(path, id string) error {
	return m.loadErr
}

func TestServer_LoadAdapterHandler(t *testing.T) {
	s := &Server{
		Engine: &mockAdapterEngine{},
	}

	// 1. Invalid method
	req := httptest.NewRequest("GET", "/v1/adapters/load", nil)
	rr := httptest.NewRecorder()
	s.LoadAdapterHandler(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", rr.Code)
	}

	// 2. Bad JSON
	req = httptest.NewRequest("POST", "/v1/adapters/load", bytes.NewBufferString("{invalid"))
	rr = httptest.NewRecorder()
	s.LoadAdapterHandler(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for bad JSON, got %d", rr.Code)
	}

	// 3. Missing path or ID
	body, _ := json.Marshal(LoadAdapterRequest{Path: "", ID: "test"})
	req = httptest.NewRequest("POST", "/v1/adapters/load", bytes.NewBuffer(body))
	rr = httptest.NewRecorder()
	s.LoadAdapterHandler(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for missing path, got %d", rr.Code)
	}

	// 4. Engine does not support adapters (nil engine)
	sNoAdapter := &Server{Engine: nil}
	body, _ = json.Marshal(LoadAdapterRequest{Path: "/path", ID: "test"})
	req = httptest.NewRequest("POST", "/v1/adapters/load", bytes.NewBuffer(body))
	rr = httptest.NewRecorder()
	sNoAdapter.LoadAdapterHandler(rr, req)
	if rr.Code != http.StatusNotImplemented {
		t.Errorf("expected 501 for engine without adapters, got %d", rr.Code)
	}

	// 5. Engine returns load error
	sErr := &Server{Engine: &mockAdapterEngine{loadErr: errors.New("failed load")}}
	req = httptest.NewRequest("POST", "/v1/adapters/load", bytes.NewBuffer(body))
	rr = httptest.NewRecorder()
	sErr.LoadAdapterHandler(rr, req)
	if rr.Code != http.StatusInternalServerError {
		t.Errorf("expected 500 on load error, got %d", rr.Code)
	}

	// 6. Success
	req = httptest.NewRequest("POST", "/v1/adapters/load", bytes.NewBuffer(body))
	rr = httptest.NewRecorder()
	s.LoadAdapterHandler(rr, req)
	if rr.Code != http.StatusCreated {
		t.Errorf("expected 201 on success, got %d", rr.Code)
	}
}
