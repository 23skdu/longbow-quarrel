package api

import (
	"encoding/json"
	"net/http"
)

// Server handles REST endpoints including orchestration health checks.
type Server struct {
	MaxMemory int64
	UsedMemory func() int64
}

type HealthResponse struct {
	Status      string `json:"status"`
	MemoryUsed  int64  `json:"memory_used"`
	MemoryTotal int64  `json:"memory_total"`
	LoadPercent int    `json:"load_percent"`
}

// HealthzEndpoint enables Kubernetes liveness/readiness orchestration probes.
func (s *Server) HealthzEndpoint(w http.ResponseWriter, r *http.Request) {
	used := s.UsedMemory()
	
	loadPercent := 0
	if s.MaxMemory > 0 {
		loadPercent = int((float64(used) / float64(s.MaxMemory)) * 100)
	}

	resp := HealthResponse{
		MemoryUsed:  used,
		MemoryTotal: s.MaxMemory,
		LoadPercent: loadPercent,
	}

	// Graceful Degradation: Fast fail new requests if OOM is imminent
	if loadPercent >= 95 {
		resp.Status = "degraded (OOM imminent)"
		w.WriteHeader(http.StatusServiceUnavailable) // 503
	} else {
		resp.Status = "ok"
		w.WriteHeader(http.StatusOK) // 200
	}

	json.NewEncoder(w).Encode(resp)
}

// Global server instance tracking memory utilization directly from the engine runtime.
var globalServer *Server

func InitServer(maxMemory int64, memCallback func() int64) {
	globalServer = &Server{
		MaxMemory:  maxMemory,
		UsedMemory: memCallback,
	}
	
	http.HandleFunc("/healthz", globalServer.HealthzEndpoint)
}
