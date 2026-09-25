package vector

import (
	"fmt"
)

// DataType defines the representation format of the vector elements.
type DataType string

const (
	TypeFloat32    DataType = "float32"
	TypeTurboQuant DataType = "turboquant"
	TypeComplex128 DataType = "complex128"
	TypeUint8      DataType = "uint8"
)

// SearchType defines the search algorithm and index topology.
type SearchType string

const (
	SearchFlatCosine SearchType = "flat_cosine"
	SearchFlatL2     SearchType = "flat_l2"
	SearchIVF        SearchType = "ivf"
	SearchHNSW       SearchType = "hnsw"
)

// DistanceMetric defines the mathematical distance or similarity function.
type DistanceMetric string

const (
	MetricDot    DistanceMetric = "dot"
	MetricCosine DistanceMetric = "cosine"
	MetricL2     DistanceMetric = "l2"
)

// SearchResult represents a retrieved candidate vector and its similarity or distance score.
type SearchResult struct {
	ID    int
	Score float32
}

// TurboQuantVector holds a compressed vector with polar quantization and 1-bit QJL residual.
type TurboQuantVector struct {
	ID       int
	Codes    []int8
	Scale    float32
	QJLBits  []byte
	QJLScale float32
	Norm     float32
}

// Uint8Vector holds an 8-bit quantized integer vector with affine scale and offset.
type Uint8Vector struct {
	ID     int
	Data   []uint8
	Scale  float32
	Offset float32
}

// BenchmarkResult stores the performance and resource metrics for a test run.
type BenchmarkResult struct {
	VectorCount      int            `json:"vector_count"`
	Dimension        int            `json:"dimension"`
	DataType         DataType       `json:"data_type"`
	SearchType       SearchType     `json:"search_type"`
	Metric           DistanceMetric `json:"metric"`
	BuildTimeMs      float64        `json:"build_time_ms"`
	QueryCount       int            `json:"query_count"`
	TotalDurationMs  float64        `json:"total_duration_ms"`
	QPS              float64        `json:"qps"`
	LatencyMeanUs    float64        `json:"latency_mean_us"`
	LatencyP50Us     float64        `json:"latency_p50_us"`
	LatencyP95Us     float64        `json:"latency_p95_us"`
	LatencyP99Us     float64        `json:"latency_p99_us"`
	MemoryAllocMB    float64        `json:"memory_alloc_mb"`
	BytesPerVector   float64        `json:"bytes_per_vector"`
	CompressionRatio float64        `json:"compression_ratio"`
	ResidualLeakKB   float64        `json:"residual_leak_kb"`
}

func (r BenchmarkResult) String() string {
	return fmt.Sprintf("N=%d D=%d [%s|%s]: QPS=%.1f Latency=%.2fμs (P95=%.2fμs) Mem=%.2fMB (%.1f B/vec)",
		r.VectorCount, r.Dimension, r.DataType, r.SearchType, r.QPS, r.LatencyMeanUs, r.LatencyP95Us, r.MemoryAllocMB, r.BytesPerVector)
}
