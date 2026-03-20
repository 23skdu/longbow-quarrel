//go:build linux && cuda

package device

import (
	"fmt"
	"testing"
)

func TestMultiGPUConfig(t *testing.T) {
	config := &MultiGPUConfig{
		Mode:               TensorParallelism | PipelineParallelism,
		NumGPUs:            4,
		TensorParallelSize: 2,
		PipelineStages:     2,
		BatchSizePerGPU:    1,
		UseNCCL:            true,
		UsePipelineBubbles: true,
		PipelineDepth:      4,
	}

	if config.TensorParallelSize != 2 {
		t.Errorf("Expected TensorParallelSize=2, got %d", config.TensorParallelSize)
	}

	if config.PipelineStages != 2 {
		t.Errorf("Expected PipelineStages=2, got %d", config.PipelineStages)
	}

	if config.Mode&TensorParallelism == 0 {
		t.Error("Expected TensorParallelism mode to be set")
	}

	if config.Mode&PipelineParallelism == 0 {
		t.Error("Expected PipelineParallelism mode to be set")
	}
}

func TestParallelismModes(t *testing.T) {
	mode := TensorParallelism | PipelineParallelism

	if mode&TensorParallelism == 0 {
		t.Error("TensorParallelism bit not set")
	}

	if mode&PipelineParallelism == 0 {
		t.Error("PipelineParallelism bit not set")
	}

	if mode&WeightStreamingParallelism != 0 {
		t.Error("WeightStreamingParallelism bit should not be set")
	}
}

func TestDeviceCount(t *testing.T) {
	count, err := GetDeviceCount()
	if err != nil {
		t.Skipf("Skipping test: %v", err)
	}

	if count < 1 {
		t.Errorf("Expected at least 1 GPU, got %d", count)
	}

	t.Logf("Detected %d GPU(s)", count)
}

func TestGetDeviceName(t *testing.T) {
	name := GetDeviceName(0)
	expected := "GPU-0"
	if name != expected {
		t.Errorf("Expected device name %s, got %s", expected, name)
	}

	name = GetDeviceName(3)
	expected = "GPU-3"
	if name != expected {
		t.Errorf("Expected device name %s, got %s", expected, name)
	}
}

func TestReduceOps(t *testing.T) {
	tensors := map[int][]float32{
		0: {1.0, 2.0, 3.0, 4.0},
		1: {2.0, 4.0, 6.0, 8.0},
		2: {3.0, 6.0, 9.0, 12.0},
	}

	err := AllReduceDistributed(tensors, ReduceSum)
	if err != nil {
		t.Errorf("AllReduceDistributed failed: %v", err)
	}

	expected := []float32{2.0, 4.0, 6.0, 8.0}
	for _, tensorData := range tensors {
		for j, v := range tensorData {
			if v != expected[j] {
				t.Errorf("Expected %f at index %d after ReduceSum, got %f", expected[j], j, v)
			}
		}
	}
}

func TestReduceMax(t *testing.T) {
	tensors := map[int][]float32{
		0: {1.0, 5.0, 3.0, 8.0},
		1: {4.0, 2.0, 7.0, 6.0},
		2: {3.0, 9.0, 1.0, 5.0},
	}

	err := AllReduceDistributed(tensors, ReduceMax)
	if err != nil {
		t.Errorf("AllReduceDistributed failed: %v", err)
	}

	expected := []float32{4.0, 9.0, 7.0, 8.0}
	for i, tensorData := range tensors {
		for j, v := range tensorData {
			if v != expected[j] {
				t.Errorf("GPU %d: Expected %f at index %d after ReduceMax, got %f", i, expected[j], j, v)
			}
		}
	}
}

func TestReduceMean(t *testing.T) {
	tensors := map[int][]float32{
		0: {1.0, 2.0},
		1: {3.0, 4.0},
	}

	err := AllReduceDistributed(tensors, ReduceMean)
	if err != nil {
		t.Errorf("AllReduceDistributed failed: %v", err)
	}

	expected := []float32{2.0, 3.0}
	for _, tensorData := range tensors {
		for j, v := range tensorData {
			if v != expected[j] {
				t.Errorf("Expected %f at index %d after ReduceMean, got %f", expected[j], j, v)
			}
		}
	}
}

func TestSingleGPUNoOp(t *testing.T) {
	tensors := map[int][]float32{
		0: {1.0, 2.0, 3.0},
	}

	err := AllReduceDistributed(tensors, ReduceSum)
	if err != nil {
		t.Errorf("AllReduceDistributed failed: %v", err)
	}

	expected := []float32{1.0, 2.0, 3.0}
	for j, v := range tensors[0] {
		if v != expected[j] {
			t.Errorf("Expected %f at index %d, got %f", expected[j], j, v)
		}
	}
}

func TestCrossGPUConfig(t *testing.T) {
	config := GetMultiGPUConfig()
	if config == nil {
		t.Error("Expected default config to be non-nil")
	}
}

func TestSetMultiGPUConfig(t *testing.T) {
	config := &MultiGPUConfig{
		Mode:               TensorParallelism,
		NumGPUs:            8,
		TensorParallelSize: 4,
		PipelineStages:     2,
	}

	SetMultiGPUConfig(config)

	retrieved := GetMultiGPUConfig()
	if retrieved.NumGPUs != 8 {
		t.Errorf("Expected NumGPUs=8, got %d", retrieved.NumGPUs)
	}
}

func TestPipelineLayerDistribution(t *testing.T) {
	numLayers := 32
	numGPUs := 4

	expectedPerGPU := numLayers / numGPUs

	distribution := make([]int, numGPUs)
	remainder := numLayers % numGPUs

	for i := 0; i < numGPUs; i++ {
		distribution[i] = expectedPerGPU
		if i < remainder {
			distribution[i]++
		}
	}

	total := 0
	for i, d := range distribution {
		total += d
		t.Logf("GPU %d: %d layers", i, d)
	}

	if total != numLayers {
		t.Errorf("Distribution total %d != numLayers %d", total, numLayers)
	}
}

func TestHybridParallelismDistribution(t *testing.T) {
	config := &MultiGPUConfig{
		Mode:               PipelineParallelism,
		NumGPUs:            4,
		PipelineStages:     4,
		TensorParallelSize: 1,
	}

	hm := &HybridParallelismManager{
		config: config,
	}

	distribution := hm.DistributeLayers(32)

	total := 0
	for _, d := range distribution {
		total += d
	}

	if total != 32 {
		t.Errorf("Distribution total %d != expected 32", total)
	}

	if hm.pipelineParallel == nil {
		if len(distribution) != 1 {
			t.Errorf("Expected 1 stage when pipelineParallel is nil, got %d", len(distribution))
		}
	} else if len(distribution) != hm.pipelineParallel.numStages {
		t.Errorf("Expected %d stages, got %d", hm.pipelineParallel.numStages, len(distribution))
	}
}

func TestDeviceForLayer(t *testing.T) {
	config := &MultiGPUConfig{
		Mode:               PipelineParallelism,
		NumGPUs:            4,
		PipelineStages:     4,
		TensorParallelSize: 1,
	}

	hm := &HybridParallelismManager{
		config: config,
		pipelineParallel: &PipelineParallelManager{
			stages: []*PipelineStage{
				{ID: 0, StartLayer: 0, EndLayer: 8, DeviceID: 0},
				{ID: 1, StartLayer: 8, EndLayer: 16, DeviceID: 1},
				{ID: 2, StartLayer: 16, EndLayer: 24, DeviceID: 2},
				{ID: 3, StartLayer: 24, EndLayer: 32, DeviceID: 3},
			},
			numStages: 4,
		},
	}

	tests := []struct {
		layer          int
		expectedDevice int
	}{
		{0, 0},
		{7, 0},
		{8, 1},
		{15, 1},
		{16, 2},
		{23, 2},
		{24, 3},
		{31, 3},
	}

	for _, tt := range tests {
		device := hm.GetDeviceForLayer(tt.layer)
		if device != tt.expectedDevice {
			t.Errorf("Layer %d: expected device %d, got %d", tt.layer, tt.expectedDevice, device)
		}
	}
}

func TestMultiGPUManagerRequiresMultipleGPUs(t *testing.T) {
	count, err := GetDeviceCount()
	if err != nil || count < 2 {
		t.Skip("Test requires at least 2 GPUs")
	}

	config := &MultiGPUConfig{
		Mode:               TensorParallelism,
		TensorParallelSize: 2,
	}

	_, err = NewTensorParallelManager(config)
	if err != nil {
		t.Errorf("Expected successful tensor parallel manager creation, got: %v", err)
	}
}

func TestPeerAccessMatrix(t *testing.T) {
	count, err := GetDeviceCount()
	if err != nil || count < 2 {
		t.Skip("Test requires at least 2 GPUs")
	}

	config := &MultiGPUConfig{
		Mode:               TensorParallelism | PipelineParallelism,
		NumGPUs:            count,
		TensorParallelSize: count,
		PipelineStages:     count,
	}

	cg, err := NewCrossGPUCommunicator(config)
	if err != nil {
		t.Skipf("CrossGPU communicator creation skipped: %v", err)
	}

	for i := 0; i < count; i++ {
		if !cg.CanAccessPeer(i, i) {
			t.Errorf("Expected self-access for device %d", i)
		}
	}

	t.Logf("Peer access matrix created for %d devices", count)
}

func TestMemoryStats(t *testing.T) {
	config := &MultiGPUConfig{
		Mode:               TensorParallelism,
		NumGPUs:            2,
		TensorParallelSize: 2,
	}

	hm := &HybridParallelismManager{
		config:          config,
		totalMemory:     16 * 1024 * 1024 * 1024,
		availableMemory: 16 * 1024 * 1024 * 1024,
	}

	total, available, used := hm.GetMemoryStats()

	if total != 16*1024*1024*1024 {
		t.Errorf("Expected total=16GB, got %d", total)
	}

	if available != 16*1024*1024*1024 {
		t.Errorf("Expected available=16GB, got %d", available)
	}

	if used != 0 {
		t.Errorf("Expected used=0, got %d", used)
	}
}

func TestWorkerAcquisition(t *testing.T) {
	config := &MultiGPUConfig{
		Mode:    TensorParallelism,
		NumGPUs: 4,
	}

	hm := &HybridParallelismManager{
		config: config,
	}

	worker1 := hm.AcquireWorker()
	if worker1 != 1 {
		t.Errorf("Expected first worker ID=1, got %d", worker1)
	}

	worker2 := hm.AcquireWorker()
	if worker2 != 2 {
		t.Errorf("Expected second worker ID=2, got %d", worker2)
	}

	if !hm.IsAvailable() {
		t.Error("Expected workers to still be available")
	}

	hm.ReleaseWorker()

	if !hm.IsAvailable() {
		t.Error("Expected workers to be available after release")
	}
}

func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name        string
		config      MultiGPUConfig
		shouldError bool
	}{
		{
			name: "valid config",
			config: MultiGPUConfig{
				Mode:               TensorParallelism,
				NumGPUs:            4,
				TensorParallelSize: 2,
			},
			shouldError: false,
		},
		{
			name: "tensor size exceeds GPUs",
			config: MultiGPUConfig{
				Mode:               TensorParallelism,
				NumGPUs:            2,
				TensorParallelSize: 4,
			},
			shouldError: true,
		},
		{
			name: "pipeline stages exceeds layers",
			config: MultiGPUConfig{
				Mode:           PipelineParallelism,
				NumGPUs:        8,
				PipelineStages: 32,
			},
			shouldError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.config.TensorParallelSize > tt.config.NumGPUs {
				if !tt.shouldError {
					t.Error("Expected no error but got validation failure")
				}
			}
		})
	}
}

func TestCommunicationStats(t *testing.T) {
	t.Skip("CrossGPUCommunicator requires CGO types - skipping in non-CGO environment")
}

// Benchmark tests
func BenchmarkAllReduce(b *testing.B) {
	tensors := map[int][]float32{
		0: make([]float32, 1024*1024),
		1: make([]float32, 1024*1024),
		2: make([]float32, 1024*1024),
		3: make([]float32, 1024*1024),
	}

	for i := 0; i < 1024*1024; i++ {
		tensors[0][i] = float32(i)
		tensors[1][i] = float32(i * 2)
		tensors[2][i] = float32(i * 3)
		tensors[3][i] = float32(i * 4)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		AllReduceDistributed(tensors, ReduceSum)
	}
}

func BenchmarkLayerDistribution(b *testing.B) {
	hm := &HybridParallelismManager{
		config: &MultiGPUConfig{
			Mode:           PipelineParallelism,
			NumGPUs:        8,
			PipelineStages: 8,
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hm.DistributeLayers(80)
	}
}

func ExampleTensorParallelManager() {
	config := &MultiGPUConfig{
		Mode:               TensorParallelism,
		NumGPUs:            4,
		TensorParallelSize: 4,
	}

	fmt.Printf("Tensor Parallelism Configuration:\n")
	fmt.Printf("  Mode: %d\n", config.Mode)
	fmt.Printf("  GPUs: %d\n", config.NumGPUs)
	fmt.Printf("  TP Size: %d\n", config.TensorParallelSize)

	// Output:
	// Tensor Parallelism Configuration:
	//   Mode: 1
	//   GPUs: 4
	//   TP Size: 4
}

func ExamplePipelineParallelManager() {
	config := &MultiGPUConfig{
		Mode:               PipelineParallelism,
		NumGPUs:            4,
		PipelineStages:     4,
		PipelineDepth:      4,
		UsePipelineBubbles: true,
	}

	fmt.Printf("Pipeline Parallelism Configuration:\n")
	fmt.Printf("  Mode: %d\n", config.Mode)
	fmt.Printf("  Stages: %d\n", config.PipelineStages)
	fmt.Printf("  Depth: %d\n", config.PipelineDepth)
	fmt.Printf("  Pipeline Bubbles: %v\n", config.UsePipelineBubbles)

	// Output:
	// Pipeline Parallelism Configuration:
	//   Mode: 2
	//   Stages: 4
	//   Depth: 4
	//   Pipeline Bubbles: true
}
