//go:build linux && amd64 && cuda && cgo

package engine

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

func TestCUDAEnginePreservesCallerConfigAndWritesActivationLog(t *testing.T) {
	dir := t.TempDir()
	path := buildTestGGUF(t, dir)
	logPath := filepath.Join(dir, "activations_cuda.json")

	cfg := config.Default()
	cfg.KVCacheSize = 64
	cfg.SeqLen = 64
	cfg.NumGPULayers = -1
	cfg.Devices = []int{0}
	cfg.ActivationLogPath = logPath
	cfg.MaxBatchSize = 16
	cfg.BlockSize = 32
	cfg.TotalBlocks = 128
	cfg.TensorParallelSize = 1
	cfg.PipelineStages = 1
	cfg.MaxMemoryMB = 1024

	engCreated, err := NewcudaEngine(path, cfg)
	if err != nil {
		t.Fatalf("failed to create CUDA engine: %v", err)
	}
	defer engCreated.Close()
	cudaEng := engCreated.(*cudaEngine)

	gotCfg := cudaEng.Config()
	if gotCfg.ActivationLogPath != logPath {
		t.Errorf("ActivationLogPath was lost: got %q, want %q", gotCfg.ActivationLogPath, logPath)
	}
	if len(gotCfg.Devices) != 1 || gotCfg.Devices[0] != 0 {
		t.Errorf("Devices was lost: got %v, want [0]", gotCfg.Devices)
	}
	if gotCfg.MaxBatchSize != 16 {
		t.Errorf("MaxBatchSize was lost: got %d, want 16", gotCfg.MaxBatchSize)
	}
	if gotCfg.BlockSize != 32 {
		t.Errorf("BlockSize was lost: got %d, want 32", gotCfg.BlockSize)
	}
	if gotCfg.TotalBlocks != 128 {
		t.Errorf("TotalBlocks was lost: got %d, want 128", gotCfg.TotalBlocks)
	}
	if gotCfg.TensorParallelSize != 1 {
		t.Errorf("TensorParallelSize was lost: got %d, want 1", gotCfg.TensorParallelSize)
	}
	if gotCfg.PipelineStages != 1 {
		t.Errorf("PipelineStages was lost: got %d, want 1", gotCfg.PipelineStages)
	}
	if gotCfg.MaxMemoryMB != 1024 {
		t.Errorf("MaxMemoryMB was lost: got %d, want 1024", gotCfg.MaxMemoryMB)
	}

	// Run a forward pass and verify that the activation log is written and valid JSON
	batch := &BatchDescriptor{
		Sequences:   []*Sequence{{ID: 1, Tokens: []int{1, 2}, PromptLen: 2}},
		Tokens:      []int{1, 2},
		Offsets:     []int{0},
		ContextLens: []int{0},
		TokenToSeq:  []int{0, 0},
	}
	results, err := cudaEng.ForwardBatch(batch)
	if err != nil {
		t.Fatalf("ForwardBatch failed: %v", err)
	}
	defer results[0].Free()

	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("expected activation log at %s, but read failed: %v", logPath, err)
	}
	var actLog ActivationLog
	if err := json.Unmarshal(data, &actLog); err != nil {
		t.Fatalf("activation log is not valid JSON: %v", err)
	}
	if len(actLog.Embedding) == 0 {
		t.Errorf("activation log has empty embedding")
	}
	if len(actLog.Layers) == 0 {
		t.Errorf("activation log has no layers")
	}

	// Compare with CPU engine config to ensure CPU and CUDA keep the same caller fields
	cpuCreated, err := NewCPUEngine(path, cfg)
	if err != nil {
		t.Fatalf("failed to create CPU engine: %v", err)
	}
	defer cpuCreated.Close()
	cpuCfg := cpuCreated.Config()

	if gotCfg.ActivationLogPath != cpuCfg.ActivationLogPath {
		t.Errorf("config mismatch ActivationLogPath: cuda=%q, cpu=%q", gotCfg.ActivationLogPath, cpuCfg.ActivationLogPath)
	}
	if gotCfg.MaxBatchSize != cpuCfg.MaxBatchSize {
		t.Errorf("config mismatch MaxBatchSize: cuda=%d, cpu=%d", gotCfg.MaxBatchSize, cpuCfg.MaxBatchSize)
	}
	if gotCfg.BlockSize != cpuCfg.BlockSize {
		t.Errorf("config mismatch BlockSize: cuda=%d, cpu=%d", gotCfg.BlockSize, cpuCfg.BlockSize)
	}
	if gotCfg.TotalBlocks != cpuCfg.TotalBlocks {
		t.Errorf("config mismatch TotalBlocks: cuda=%d, cpu=%d", gotCfg.TotalBlocks, cpuCfg.TotalBlocks)
	}
}
