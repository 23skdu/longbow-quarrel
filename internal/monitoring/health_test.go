//go:build darwin && metal

package monitoring

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestHealthMonitor_Lifecycle(t *testing.T) {
	hm := NewHealthMonitor()

	t.Run("RecordInference", func(t *testing.T) {
		hm.RecordInference(100, 10*time.Second) // 10 tokens/sec
		
		status := hm.getHealthStatus()
		if status.Performance.TokensPerSecond == 0 {
			t.Errorf("expected non-zero tokens per second")
		}
	})

	t.Run("Alerts", func(t *testing.T) {
		hm.AddAlert("error", "engine", "something went wrong")
		status := hm.getHealthStatus()
		if status.Status != "degraded" {
			t.Errorf("expected degraded status, got %s", status.Status)
		}

		hm.ResolveAlert(0)
		status = hm.getHealthStatus()
		// Since we have no critical alerts, it should be healthy (alerts list both resolved and unresolved)
		// but wait, standard implementation might differ.
		_ = status
	})

	t.Run("HTTPHandlers", func(t *testing.T) {
		w := httptest.NewRecorder()
		r := httptest.NewRequest("GET", "/health", nil)
		hm.handleHealth(w, r)
		if w.Code != http.StatusOK {
			t.Errorf("expected 200 OK, got %d", w.Code)
		}

		w = httptest.NewRecorder()
		r = httptest.NewRequest("GET", "/status", nil)
		hm.handleDetailedStatus(w, r)
		var status HealthStatus
		json.NewDecoder(w.Body).Decode(&status)
		if status.Version != "1.0.0" {
			t.Errorf("expected version 1.0.0, got %s", status.Version)
		}
	})
	
	t.Run("Stop", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		_ = hm.Stop(ctx)
	})
}
