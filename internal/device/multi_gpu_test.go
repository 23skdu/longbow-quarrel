//go:build linux && amd64 && cuda && cgo

package device

import (
	"fmt"
	"sync"
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
	if name == "" {
		t.Errorf("Expected non-empty device name, got empty")
	} else {
		t.Logf("Device 0 name: %s", name)
	}

	name = GetDeviceName(999)
	if name != "Unknown" && name != "GPU-999" {
		t.Logf("Device 999 name: %s", name)
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
	for b.Loop() {
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
	for b.Loop() {
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

func Test1F1BDoubleBuffering(t *testing.T) {
	stage := &PipelineStage{
		ID:         0,
		StartLayer: 0,
		EndLayer:   8,
		DeviceID:   0,
	}

	dim := 128
	stage.InitDoubleBuffers(dim)

	if len(stage.HostStagingIn[0]) != dim || len(stage.HostStagingIn[1]) != dim {
		t.Fatalf("Expected HostStagingIn to be of size %d, got %d and %d",
			dim, len(stage.HostStagingIn[0]), len(stage.HostStagingIn[1]))
	}

	// Test slot locking
	stage.AcquireSlot(0)
	stage.ReleaseSlot(0)

	stage.AcquireSlot(1)
	stage.ReleaseSlot(1)

	// Test slot indexing modulo 2
	stage.AcquireSlot(2)
	stage.ReleaseSlot(2)
}

func Test1F1BPipelineExecution(t *testing.T) {
	ShutdownMultiGPU()
	defer ShutdownMultiGPU()

	cfg := &MultiGPUConfig{
		Mode:           PipelineParallelism,
		NumGPUs:        1,
		PipelineStages: 2,
		PipelineDepth:  4,
	}

	pp, err := NewPipelineParallelManager(cfg, 16)
	if err != nil {
		t.Fatalf("Failed to create PipelineParallelManager: %v", err)
	}

	// 8 micro-batches with dim 4
	numBatches := 8
	dim := 4
	microBatches := make([][]float32, numBatches)
	for i := 0; i < numBatches; i++ {
		microBatches[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			microBatches[i][j] = float32(i*10 + j + 1)
		}
	}

	// Track slots used to verify double-buffer alternating pattern
	var slotHistory sync.Map
	stageFn := func(stage *PipelineStage, microBatchID int, slot int, input []float32) ([]float32, error) {
		expectedSlot := microBatchID % 2
		if slot != expectedSlot {
			t.Errorf("Microbatch %d: expected slot %d, got %d", microBatchID, expectedSlot, slot)
		}
		slotHistory.Store(fmt.Sprintf("%d-%d", stage.ID, microBatchID), slot)

		output := make([]float32, len(input))
		for k, v := range input {
			// Stage 0 adds 100, Stage 1 adds 200
			output[k] = v + float32((stage.ID+1)*100)
		}
		return output, nil
	}

	results, err := pp.ForwardMicroBatches1F1B(microBatches, stageFn)
	if err != nil {
		t.Fatalf("ForwardMicroBatches1F1B failed: %v", err)
	}

	if len(results) != numBatches {
		t.Fatalf("Expected %d results, got %d", numBatches, len(results))
	}

	for i := 0; i < numBatches; i++ {
		for j := 0; j < dim; j++ {
			// Initial: i*10 + j + 1
			// Stage 0: + 100
			// Stage 1: + 200
			// Total: i*10 + j + 1 + 300
			expected := float32(i*10 + j + 1 + 300)
			if results[i][j] != expected {
				t.Errorf("Batch %d, element %d: expected %f, got %f", i, j, expected, results[i][j])
			}
		}
	}

	// Verify forward pass counts
	fwdCount := pp.GetFwdPassCount()
	if fwdCount < int64(numBatches) {
		t.Errorf("Expected fwdCount >= %d, got %d", numBatches, fwdCount)
	}

	// Verify direct ForwardStage execution
	stageOut, err := pp.ForwardStage(pp.stages[0], []float32{1.0, 2.0}, 0)
	if err != nil {
		t.Errorf("ForwardStage failed: %v", err)
	}
	if len(stageOut) != 2 {
		t.Errorf("Expected 2 outputs from ForwardStage, got %d", len(stageOut))
	}
}

func Test1F1BSingleStageExecution(t *testing.T) {
	ShutdownMultiGPU()
	defer ShutdownMultiGPU()

	cfg := &MultiGPUConfig{
		Mode:           PipelineParallelism,
		NumGPUs:        1,
		PipelineStages: 1,
		PipelineDepth:  2,
	}

	pp, err := NewPipelineParallelManager(cfg, 8)
	if err != nil {
		t.Fatalf("Failed to create PipelineParallelManager: %v", err)
	}

	input := [][]float32{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
	}

	results, err := pp.ForwardMicroBatches1F1B(input, nil)
	if err != nil {
		t.Fatalf("Single stage 1F1B failed: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(results))
	}

	for i := range input {
		for j := range input[i] {
			if results[i][j] != input[i][j] {
				t.Errorf("Mismatch at [%d][%d]: expected %f, got %f", i, j, input[i][j], results[i][j])
			}
		}
	}

	// Also test ForwardPass wrapper
	singleOut, err := pp.ForwardPass(0, input[0])
	if err != nil {
		t.Fatalf("ForwardPass failed: %v", err)
	}
	if len(singleOut) != 2 || singleOut[0] != 1.0 {
		t.Errorf("ForwardPass unexpected output: %v", singleOut)
	}
}

func TestCrossGPUStagingFallback(t *testing.T) {
	ShutdownMultiGPU()
	defer ShutdownMultiGPU()

	cfg := &MultiGPUConfig{
		Mode:    PipelineParallelism | TensorParallelism,
		NumGPUs: 1,
	}

	cg, err := NewCrossGPUCommunicator(cfg)
	if err != nil {
		t.Fatalf("NewCrossGPUCommunicator failed: %v", err)
	}

	// Test peer buffer allocation on device 0
	pm, err := cg.AllocatePeerStagingBuffer(0, 0, 1024)
	if err != nil {
		t.Fatalf("AllocatePeerStagingBuffer failed: %v", err)
	}
	if pm == nil || !pm.isValid || pm.size != 1024 {
		t.Errorf("Unexpected PeerMemory: %+v", pm)
	}

	// Test cached allocation returns existing
	pm2, err := cg.AllocatePeerStagingBuffer(0, 0, 512)
	if err != nil || pm2 != pm {
		t.Errorf("Expected cached buffer reuse, got %v", pm2)
	}

	// Test activation transfer
	err = cg.TransferActivations(0, 0, []float32{1.0, 2.0, 3.0, 4.0}, 0)
	if err != nil {
		t.Errorf("TransferActivations self-device failed: %v", err)
	}

	ops, sent, recv := cg.GetStats()
	t.Logf("Stats: ops=%d, sent=%d, recv=%d", ops, sent, recv)
}

func TestMultiGPUConfigDevices(t *testing.T) {
	cfg := &MultiGPUConfig{
		Mode:               TensorParallelism | PipelineParallelism,
		Devices:            []int{0, 1},
		TensorParallelSize: 2,
		PipelineStages:     2,
	}

	if len(cfg.Devices) != 2 || cfg.Devices[0] != 0 || cfg.Devices[1] != 1 {
		t.Errorf("Devices not preserved: %v", cfg.Devices)
	}
}

func TestTensorParallelSingleGPU(t *testing.T) {
	ShutdownMultiGPU()
	defer ShutdownMultiGPU()

	cfg := &MultiGPUConfig{
		Mode:               TensorParallelism,
		TensorParallelSize: 1,
		Devices:            []int{0},
	}

	tp, err := NewTensorParallelManager(cfg)
	if err != nil {
		t.Fatalf("Expected TP=1 on single GPU to succeed, got: %v", err)
	}

	if tp.GetWorldSize() != 1 {
		t.Errorf("Expected WorldSize=1, got %d", tp.GetWorldSize())
	}

	data := []float32{10.0, 20.0, 30.0}
	err = tp.AllReduce(data, len(data))
	if err != nil {
		t.Errorf("AllReduce on WorldSize=1 failed: %v", err)
	}
	if data[0] != 10.0 {
		t.Errorf("Expected data[0]=10.0, got %f", data[0])
	}
}
