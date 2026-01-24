//go:build darwin && metal

package monitoring

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"sync"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// HealthStatus represents the health status of the system
type HealthStatus struct {
	Status      string          `json:"status"`
	Timestamp   time.Time       `json:"timestamp"`
	Version     string          `json:"version"`
	Uptime      time.Duration   `json:"uptime"`
	System      SystemInfo      `json:"system"`
	Engine      EngineInfo      `json:"engine"`
	Performance PerformanceInfo `json:"performance"`
	Alerts      []Alert         `json:"alerts"`
}

// SystemInfo contains system-level information
type SystemInfo struct {
	GoVersion      string  `json:"go_version"`
	OS             string  `json:"os"`
	Arch           string  `json:"arch"`
	NumCPU         int     `json:"num_cpu"`
	MemoryMB       int     `json:"memory_mb"`
	MemoryUsedMB   int     `json:"memory_used_mb"`
	MemoryUsagePct float64 `json:"memory_usage_pct"`
}

// EngineInfo contains engine-specific information
type EngineInfo struct {
	ModelLoaded     bool    `json:"model_loaded"`
	ModelPath       string  `json:"model_path"`
	ModelSize       int64   `json:"model_size"`
	KVCacheSize     int     `json:"kv_cache_size"`
	KVCacheUsed     int     `json:"kv_cache_used"`
	KVCacheUsagePct float64 `json:"kv_cache_usage_pct"`
	NumLayers       int     `json:"num_layers"`
	NumHeads        int     `json:"num_heads"`
	ContextLength   int     `json:"context_length"`
	GPUMemoryMB     int64   `json:"gpu_memory_mb"`
}

// PerformanceInfo contains performance metrics
type PerformanceInfo struct {
	TokensPerSecond float64   `json:"tokens_per_second"`
	AvgLatencyMs    float64   `json:"avg_latency_ms"`
	P95LatencyMs    float64   `json:"p95_latency_ms"`
	ErrorRate       float64   `json:"error_rate"`
	NanCount        int       `json:"nan_count"`
	LastInference   time.Time `json:"last_inference"`
}

// Alert represents a system alert
type Alert struct {
	Level      string     `json:"level"`     // info, warning, error, critical
	Component  string     `json:"component"` // engine, gpu, memory, system
	Message    string     `json:"message"`
	Timestamp  time.Time  `json:"timestamp"`
	Resolved   bool       `json:"resolved"`
	ResolvedAt *time.Time `json:"resolved_at,omitempty"`
}

// HealthMonitor monitors system health
type HealthMonitor struct {
	startTime     time.Time
	server        *http.Server
	mu            sync.RWMutex
	alerts        []Alert
	lastInference time.Time
	perfHistory   []PerfPoint
	healthStatus  HealthStatus
}

// PerfPoint represents a performance data point
type PerfPoint struct {
	Timestamp time.Time
	Tokens    int
	Duration  time.Duration
}

// NewHealthMonitor creates a new health monitor
func NewHealthMonitor() *HealthMonitor {
	return &HealthMonitor{
		startTime:   time.Now(),
		alerts:      make([]Alert, 0),
		perfHistory: make([]PerfPoint, 0),
	}
}

// Start begins health monitoring
func (hm *HealthMonitor) Start(addr string) error {
	mux := http.NewServeMux()

	// Health endpoint
	mux.HandleFunc("/health", hm.handleHealth)
	mux.HandleFunc("/healthz", hm.handleHealth) // Kubernetes compatibility

	// Metrics endpoint (Prometheus)
	mux.Handle("/metrics", promhttp.Handler())

	// Detailed status endpoint
	mux.HandleFunc("/status", hm.handleDetailedStatus)

	// Admin endpoints
	mux.HandleFunc("/admin/alerts", hm.handleAlerts)
	mux.HandleFunc("/admin/clear-alerts", hm.handleClearAlerts)

	hm.server = &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	log.Printf("Health monitor starting on %s", addr)
	return hm.server.ListenAndServe()
}

// Stop stops health monitoring
func (hm *HealthMonitor) Stop(ctx context.Context) error {
	if hm.server != nil {
		return hm.server.Shutdown(ctx)
	}
	return nil
}

// RecordInference records an inference event for performance monitoring
func (hm *HealthMonitor) RecordInference(tokens int, duration time.Duration) {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	now := time.Now()
	hm.lastInference = now

	// Record to Prometheus
	metrics.RecordInference(tokens, duration)

	// Add to performance history
	point := PerfPoint{
		Timestamp: now,
		Tokens:    tokens,
		Duration:  duration,
	}

	hm.perfHistory = append(hm.perfHistory, point)

	// Keep only last 1000 points
	if len(hm.perfHistory) > 1000 {
		hm.perfHistory = hm.perfHistory[1:]
	}

	// Check for performance alerts
	hm.checkPerformanceAlerts(point)
}

// RecordGPUMemory records GPU memory usage
func (hm *HealthMonitor) RecordGPUMemory(bytes int64) {
	metrics.RecordGPUMemory(bytes)
	hm.checkGPUMemoryAlerts(bytes)
}

// RecordKernelDuration records kernel execution time
func (hm *HealthMonitor) RecordKernelDuration(name string, duration time.Duration) {
	metrics.RecordKernelDuration(name, duration)
	hm.checkKernelAlerts(name, duration)
}

// AddAlert adds a new alert
func (hm *HealthMonitor) AddAlert(level, component, message string) {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	alert := Alert{
		Level:     level,
		Component: component,
		Message:   message,
		Timestamp: time.Now(),
		Resolved:  false,
	}

	hm.alerts = append(hm.alerts, alert)

	// Keep only last 100 alerts
	if len(hm.alerts) > 100 {
		hm.alerts = hm.alerts[1:]
	}

	log.Printf("ALERT [%s/%s]: %s", level, component, message)
}

// ResolveAlert resolves an alert
func (hm *HealthMonitor) ResolveAlert(index int) {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	if index >= 0 && index < len(hm.alerts) {
		now := time.Now()
		hm.alerts[index].Resolved = true
		hm.alerts[index].ResolvedAt = &now
	}
}

// HTTP Handlers

func (hm *HealthMonitor) handleHealth(w http.ResponseWriter, r *http.Request) {
	status := hm.getHealthStatus()

	if status.Status == "healthy" {
		w.WriteHeader(http.StatusOK)
	} else {
		w.WriteHeader(http.StatusServiceUnavailable)
	}

	json.NewEncoder(w).Encode(map[string]string{
		"status":    status.Status,
		"timestamp": status.Timestamp.Format(time.RFC3339),
	})
}

func (hm *HealthMonitor) handleDetailedStatus(w http.ResponseWriter, r *http.Request) {
	status := hm.getHealthStatus()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (hm *HealthMonitor) handleAlerts(w http.ResponseWriter, r *http.Request) {
	hm.mu.RLock()
	alerts := make([]Alert, len(hm.alerts))
	copy(alerts, hm.alerts)
	hm.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(alerts)
}

func (hm *HealthMonitor) handleClearAlerts(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	hm.mu.Lock()
	hm.alerts = hm.alerts[:0] // Clear all alerts
	hm.mu.Unlock()

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"message": "alerts cleared"})
}

// Health status calculation

func (hm *HealthMonitor) getHealthStatus() HealthStatus {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	status := "healthy"

	// Check for critical alerts
	for _, alert := range hm.alerts {
		if alert.Level == "critical" && !alert.Resolved {
			status = "critical"
			break
		} else if alert.Level == "error" && !alert.Resolved {
			status = "degraded"
		}
	}

	// Calculate performance metrics
	perfInfo := hm.calculatePerformanceInfo()

	// Get system info
	sysInfo := hm.getSystemInfo()

	// Get engine info (mock for now)
	engineInfo := EngineInfo{
		ModelLoaded: true,
		KVCacheSize: 22,
		GPUMemoryMB: 1024, // Mock value
	}

	return HealthStatus{
		Status:      status,
		Timestamp:   time.Now(),
		Version:     "1.0.0",
		Uptime:      time.Since(hm.startTime),
		System:      sysInfo,
		Engine:      engineInfo,
		Performance: perfInfo,
		Alerts:      hm.alerts,
	}
}

func (hm *HealthMonitor) getSystemInfo() SystemInfo {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return SystemInfo{
		GoVersion:      runtime.Version(),
		OS:             runtime.GOOS,
		Arch:           runtime.GOARCH,
		NumCPU:         runtime.NumCPU(),
		MemoryMB:       int(m.Sys / 1024 / 1024),
		MemoryUsedMB:   int(m.Alloc / 1024 / 1024),
		MemoryUsagePct: float64(m.Alloc) / float64(m.Sys) * 100,
	}
}

func (hm *HealthMonitor) calculatePerformanceInfo() PerformanceInfo {
	if len(hm.perfHistory) == 0 {
		return PerformanceInfo{
			LastInference: hm.lastInference,
		}
	}

	var totalTokens int
	var totalDuration time.Duration
	var latencies []float64
	errorCount := 0

	for _, point := range hm.perfHistory {
		totalTokens += point.Tokens
		totalDuration += point.Duration

		latencyMs := float64(point.Duration.Nanoseconds()) / 1e6
		latencies = append(latencies, latencyMs)
	}

	// Calculate percentiles
	if len(latencies) > 0 {
		// Simple percentile calculation
		for i := range latencies {
			for j := i + 1; j < len(latencies); j++ {
				if latencies[i] > latencies[j] {
					latencies[i], latencies[j] = latencies[j], latencies[i]
				}
			}
		}

		p95Index := int(float64(len(latencies)) * 0.95)
		if p95Index >= len(latencies) {
			p95Index = len(latencies) - 1
		}

		avgLatencyMs := float64(totalDuration.Nanoseconds()) / float64(len(hm.perfHistory)) / 1e6
		tokensPerSecond := float64(totalTokens) / totalDuration.Seconds()

		return PerformanceInfo{
			TokensPerSecond: tokensPerSecond,
			AvgLatencyMs:    avgLatencyMs,
			P95LatencyMs:    latencies[p95Index],
			ErrorRate:       float64(errorCount) / float64(len(hm.perfHistory)),
			NanCount:        0, // Would be calculated from NaN detection
			LastInference:   hm.lastInference,
		}
	}

	return PerformanceInfo{
		LastInference: hm.lastInference,
	}
}

// Alert checking functions

func (hm *HealthMonitor) checkPerformanceAlerts(point PerfPoint) {
	tokensPerSecond := float64(point.Tokens) / point.Duration.Seconds()

	if tokensPerSecond < 1.0 {
		hm.AddAlert("warning", "performance",
			fmt.Sprintf("Low throughput: %.2f tokens/sec", tokensPerSecond))
	}

	latencyMs := float64(point.Duration.Nanoseconds()) / 1e6
	if latencyMs > 5000 { // 5 seconds
		hm.AddAlert("error", "performance",
			fmt.Sprintf("High latency: %.2f ms", latencyMs))
	}
}

func (hm *HealthMonitor) checkGPUMemoryAlerts(bytes int64) {
	memoryMB := bytes / (1024 * 1024)

	if memoryMB > 2048 { // 2GB
		hm.AddAlert("warning", "gpu",
			fmt.Sprintf("High GPU memory usage: %d MB", memoryMB))
	}
}

func (hm *HealthMonitor) checkKernelAlerts(name string, duration time.Duration) {
	durationMs := float64(duration.Nanoseconds()) / 1e6

	if durationMs > 1000 { // 1 second
		hm.AddAlert("warning", "kernel",
			fmt.Sprintf("Slow kernel %s: %.2f ms", name, durationMs))
	}
}
