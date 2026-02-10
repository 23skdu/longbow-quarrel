//go:build ignore

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

type BenchmarkConfig struct {
	BaseURL      string
	NumClients   int
	NumRequests  int
	Duration     time.Duration
	APIKey       string
	OutputFormat string
}

type BenchmarkResult struct {
	TotalRequests  int64
	SuccessfulReqs int64
	FailedReqs     int64
	TotalDuration  time.Duration
	MinLatency     time.Duration
	MaxLatency     time.Duration
	AvgLatency     time.Duration
	RequestsPerSec float64
	Percentiles    map[string]time.Duration
}

type LatencySample struct {
	Latency time.Duration
	Success bool
}

var (
	config       BenchmarkConfig
	results      BenchmarkResult
	latencies    []LatencySample
	latencyMutex sync.Mutex
	testStart    time.Time
)

func init() {
	flag.StringVar(&config.BaseURL, "url", "http://localhost:8080", "Base URL for the API")
	flag.IntVar(&config.NumClients, "clients", 10, "Number of concurrent clients")
	flag.IntVar(&config.NumRequests, "requests", 100, "Number of requests per client")
	flag.DurationVar(&config.Duration, "duration", time.Minute, "Maximum test duration")
	flag.StringVar(&config.APIKey, "api-key", "", "API key for authentication")
	flag.StringVar(&config.OutputFormat, "format", "text", "Output format (text/json)")
	testStart = time.Now()
}

func main() {
	flag.Parse()

	fmt.Println("Longbow-Quarrel WebUI Load Benchmark")
	fmt.Println("=====================================")
	fmt.Printf("Base URL:     %s\n", config.BaseURL)
	fmt.Printf("Clients:      %d\n", config.NumClients)
	fmt.Printf("Requests:     %d per client\n", config.NumRequests)
	fmt.Printf("Duration:     %v max\n", config.Duration)
	fmt.Printf("API Key:      %s\n", maskAPIKey(config.APIKey))
	fmt.Println()

	startTime := time.Now()

	var wg sync.WaitGroup
	clientChan := make(chan int, config.NumClients)

	for i := 0; i < config.NumClients; i++ {
		wg.Add(1)
		go func(clientID int) {
			defer wg.Done()
			runClient(clientID, clientChan)
		}(i)
	}

	clientChan <- 1
	wg.Wait()

	elapsed := time.Since(startTime)
	results.TotalDuration = elapsed
	results.RequestsPerSec = float64(results.TotalRequests) / elapsed.Seconds()

	calculatePercentiles()

	printResults()

	saveResults("benchmark_results.json")
}

func runClient(clientID int, clientChan chan int) {
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	for i := 0; i < config.NumRequests; i++ {
		select {
		case <-clientChan:
			return
		default:
		}

		if time.Since(testStart) > config.Duration {
			return
		}

		latency := benchmarkEndpoint(client, "/api/models")
		recordResult(latency, true)

		if i%10 == 0 {
			select {
			case clientChan <- 1:
			default:
			}
		}
	}
}

func benchmarkEndpoint(client *http.Client, endpoint string) time.Duration {
	req, err := http.NewRequest(http.MethodGet, config.BaseURL+endpoint, nil)
	if err != nil {
		return -1
	}

	if config.APIKey != "" {
		req.Header.Set("Authorization", "ApiKey "+config.APIKey)
	}

	start := time.Now()
	resp, err := client.Do(req)
	latency := time.Since(start)

	if err != nil {
		atomic.AddInt64(&results.FailedReqs, 1)
		atomic.AddInt64(&results.TotalRequests, 1)
		return latency
	}
	defer resp.Body.Close()

	io.Copy(io.Discard, resp.Body)

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		atomic.AddInt64(&results.SuccessfulReqs, 1)
	} else {
		atomic.AddInt64(&results.FailedReqs, 1)
	}

	atomic.AddInt64(&results.TotalRequests, 1)
	return latency
}

func recordResult(latency time.Duration, success bool) {
	latencyMutex.Lock()
	defer latencyMutex.Unlock()

	latencies = append(latencies, LatencySample{
		Latency: latency,
		Success: success,
	})

	if results.MinLatency == 0 || latency < results.MinLatency {
		results.MinLatency = latency
	}
	if latency > results.MaxLatency {
		results.MaxLatency = latency
	}
}

func calculatePercentiles() {
	var successfulLatencies []time.Duration
	for _, sample := range latencies {
		if sample.Success {
			successfulLatencies = append(successfulLatencies, sample.Latency)
		}
	}

	if len(successfulLatencies) == 0 {
		return
	}

	results.Percentiles = make(map[string]time.Duration)
	percentileValues := []float64{50, 75, 90, 95, 99}

	for _, p := range percentileValues {
		index := int(float64(len(successfulLatencies)) * p / 100)
		if index >= len(successfulLatencies) {
			index = len(successfulLatencies) - 1
		}
		results.Percentiles[fmt.Sprintf("p%d", int(p))] = successfulLatencies[index]
	}

	var total time.Duration
	for _, l := range successfulLatencies {
		total += l
	}
	results.AvgLatency = total / time.Duration(len(successfulLatencies))
}

func printResults() {
	if config.OutputFormat == "json" {
		printJSON()
		return
	}

	fmt.Println("\nBenchmark Results")
	fmt.Println("===================")
	fmt.Printf("Total Requests:     %d\n", results.TotalRequests)
	fmt.Printf("Successful:         %d\n", results.SuccessfulReqs)
	fmt.Printf("Failed:             %d\n", results.FailedReqs)
	fmt.Printf("Success Rate:       %.2f%%\n", float64(results.SuccessfulReqs)/float64(results.TotalRequests)*100)
	fmt.Printf("Total Duration:     %v\n", results.TotalDuration)
	fmt.Printf("Throughput:         %.2f req/s\n", results.RequestsPerSec)
	fmt.Println()
	fmt.Println("Latency Statistics:")
	fmt.Printf("  Min:    %v\n", results.MinLatency)
	fmt.Printf("  Avg:    %v\n", results.AvgLatency)
	fmt.Printf("  Max:    %v\n", results.MaxLatency)
	fmt.Println()
	fmt.Println("Percentiles:")
	for _, p := range []string{"p50", "p75", "p90", "p95", "p99"} {
		if lat, ok := results.Percentiles[p]; ok {
			fmt.Printf("  %s:   %v\n", p, lat)
		}
	}
}

func printJSON() {
	data, _ := json.MarshalIndent(map[string]interface{}{
		"total_requests":   results.TotalRequests,
		"successful":       results.SuccessfulReqs,
		"failed":           results.FailedReqs,
		"success_rate":     float64(results.SuccessfulReqs) / float64(results.TotalRequests) * 100,
		"duration_seconds": results.TotalDuration.Seconds(),
		"requests_per_sec": results.RequestsPerSec,
		"latency": map[string]interface{}{
			"min": results.MinLatency.String(),
			"avg": results.AvgLatency.String(),
			"max": results.MaxLatency.String(),
			"p50": results.Percentiles["p50"].String(),
			"p75": results.Percentiles["p75"].String(),
			"p90": results.Percentiles["p90"].String(),
			"p95": results.Percentiles["p95"].String(),
			"p99": results.Percentiles["p99"].String(),
		},
	}, "", "  ")
	fmt.Println(string(data))
}

func saveResults(filename string) {
	data, _ := json.MarshalIndent(map[string]interface{}{
		"config": map[string]interface{}{
			"base_url":     config.BaseURL,
			"num_clients":  config.NumClients,
			"num_requests": config.NumRequests,
			"duration":     config.Duration.String(),
			"api_key":      maskAPIKey(config.APIKey),
		},
		"results": map[string]interface{}{
			"total_requests":   results.TotalRequests,
			"successful":       results.SuccessfulReqs,
			"failed":           results.FailedReqs,
			"success_rate":     float64(results.SuccessfulReqs) / float64(results.TotalRequests) * 100,
			"duration_seconds": results.TotalDuration.Seconds(),
			"requests_per_sec": results.RequestsPerSec,
			"latency_min":      results.MinLatency.String(),
			"latency_avg":      results.AvgLatency.String(),
			"latency_max":      results.MaxLatency.String(),
			"percentiles":      results.Percentiles,
		},
		"timestamp": time.Now().Format(time.RFC3339),
	}, "", "  ")

	os.WriteFile(filename, data, 0644)
	fmt.Printf("\nResults saved to %s\n", filename)
}

func maskAPIKey(key string) string {
	if len(key) <= 8 {
		return "****"
	}
	return key[:4] + "****" + key[len(key)-4:]
}

func generateTestPrompt() string {
	prompts := []string{
		"Explain quantum computing in simple terms.",
		"Write a Python function to calculate factorial.",
		"Translate 'Hello, world!' to Spanish.",
		"What are the benefits of exercise?",
		"Summarize the plot of Romeo and Juliet.",
	}
	return prompts[time.Now().Unix()%int64(len(prompts))]
}
