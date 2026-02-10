//go:build webui

package handlers

import (
	"encoding/json"
	"net/http"
	"runtime"
	"time"
)

const Version = "0.0.1"

type HealthStatus struct {
	Status    string            `json:"status"`
	Timestamp time.Time         `json:"timestamp"`
	Version   string            `json:"version"`
	Uptime    string            `json:"uptime"`
	Checks    map[string]Status `json:"checks"`
}

type Status struct {
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
}

type VersionInfo struct {
	Version   string `json:"version"`
	Commit    string `json:"commit,omitempty"`
	GoVersion string `json:"go_version"`
}

var (
	startTime = time.Now()
)

func HealthHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		status := HealthStatus{
			Status:    "healthy",
			Timestamp: time.Now(),
			Version:   Version,
			Uptime:    formatDuration(time.Since(startTime)),
			Checks: map[string]Status{
				"server": {Status: "healthy"},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(status)
	}
}

func HealthzHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK\n"))
	}
}

func ReadyzHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ready := true
		checks := make(map[string]Status)

		checks["memory"] = checkMemory()
		checks["goroutines"] = checkGoroutines()

		for _, check := range checks {
			if check.Status != "healthy" {
				ready = false
				break
			}
		}

		if ready {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("Ready\n"))
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"status": "not ready",
				"checks": checks,
			})
		}
	}
}

func VersionHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		info := VersionInfo{
			Version:   Version,
			Commit:    "",
			GoVersion: runtime.Version(),
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(info)
	}
}

func checkMemory() Status {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	if m.Alloc > 1024*1024*1024 {
		return Status{
			Status:  "warning",
			Message: "High memory usage",
		}
	}

	return Status{Status: "healthy"}
}

func checkGoroutines() Status {
	numGoroutines := runtime.NumGoroutine()

	if numGoroutines > 10000 {
		return Status{
			Status:  "warning",
			Message: "High number of goroutines",
		}
	}

	return Status{Status: "healthy"}
}

func formatDuration(d time.Duration) string {
	days := int(d.Hours()) / 24
	hours := int(d.Hours()) % 24
	minutes := int(d.Minutes()) % 60
	seconds := int(d.Seconds()) % 60

	result := ""
	if days > 0 {
		result += formatInt(days) + "d"
	}
	if hours > 0 {
		if result != "" {
			result += " "
		}
		result += formatInt(hours) + "h"
	}
	if minutes > 0 {
		if result != "" {
			result += " "
		}
		result += formatInt(minutes) + "m"
	}
	if seconds > 0 || result == "" {
		if result != "" {
			result += " "
		}
		result += formatInt(seconds) + "s"
	}

	return result
}

func formatInt(n int) string {
	if n == 0 {
		return "0"
	}
	result := ""
	for n > 0 {
		result = string(rune('0'+n%10)) + result
		n /= 10
	}
	return result
}
