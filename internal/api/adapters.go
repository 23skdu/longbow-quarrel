package api

import (
	"encoding/json"
	"net/http"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/telemetry"
)

type LoadAdapterRequest struct {
	Path string `json:"path"`
	ID   string `json:"id"`
}

type AdapterInfo struct {
	ID   string `json:"id"`
	Path string `json:"path,omitempty"`
}

type ListAdaptersResponse struct {
	Adapters []AdapterInfo `json:"adapters"`
}

// LoadAdapterHandler handles POST /v1/adapters/load
func (s *Server) LoadAdapterHandler(w http.ResponseWriter, r *http.Request) {
	ctx, span := telemetry.StartSpan(r.Context(), "LoadAdapterHandler")
	defer span.End()
	_ = ctx

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req LoadAdapterRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.Path == "" || req.ID == "" {
		http.Error(w, "path and id are required", http.StatusBadRequest)
		return
	}

	logger.Log.Info("Loading LoRA adapter", "id", req.ID, "path", req.Path)

	// We need to cast Engine to something that supports LoadAdapter
	type AdapterEngine interface {
		LoadAdapter(path, id string) error
	}

	ae, ok := s.Engine.(AdapterEngine)
	if !ok {
		http.Error(w, "Engine does not support LoRA adapters", http.StatusNotImplemented)
		return
	}

	if err := ae.LoadAdapter(req.Path, req.ID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
	if err := json.NewEncoder(w).Encode(map[string]string{"status": "loaded", "id": req.ID}); err != nil {
		logger.Log.Error("failed to encode adapter response", "error", err)
	}
}

// ListAdaptersHandler handles GET /v1/adapters/list
func (s *Server) ListAdaptersHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// This assumes the engine or LoRAManager can list adapters.
	// For now, we'll return a placeholder or implement List in Engine.
	resp := ListAdaptersResponse{
		Adapters: []AdapterInfo{},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		logger.Log.Error("failed to encode adapters list", "error", err)
	}
}
