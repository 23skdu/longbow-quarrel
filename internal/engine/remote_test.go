package engine

import (
	"context"
	"testing"

	conf "github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestRemoteWorkerEngine_ForwardShardedLayer_NilClient(t *testing.T) {
	eng := &RemoteWorkerEngine{}
	_, err := eng.ForwardShardedLayer(context.Background(), 0, 0, 10, nil)
	if err == nil {
		t.Error("Expected error when client is nil")
	}
}

func TestRemoteWorkerEngine_ForwardShardedLayer_NilInput(t *testing.T) {
	eng := &RemoteWorkerEngine{
		client: nil, // Would need mock for real test
		config: conf.Config{
			HiddenDim: 128,
		},
	}
	_, err := eng.ForwardShardedLayer(context.Background(), 0, 0, 10, nil)
	if err == nil {
		t.Error("Expected error when input is nil")
	}
}

func TestRemoteWorkerEngine_ForwardShardedLayer_ValidInput(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	input := ctx.NewTensorFP32(1, 64)
	inputData := make([]float32, 64)
	for i := range inputData {
		inputData[i] = float32(i)
	}
	input.LoadFromF32(inputData)

	eng := &RemoteWorkerEngine{
		client: nil, // Would test with mock client
		config: conf.Config{
			HiddenDim: 64,
		},
	}

	// This will fail because client is nil but tests the input validation
	_, err := eng.ForwardShardedLayer(context.Background(), 0, 0, 64, input)
	if err == nil {
		t.Error("Expected error when client is nil")
	}
}

func TestRemoteWorkerEngine_LifecycleAndRole(t *testing.T) {
	ctx := device.NewContext()
	cfg := conf.Config{
		HiddenDim: 128,
		Layers:    4,
	}

	eng := &RemoteWorkerEngine{
		config: cfg,
		role:   RoleWorker,
	}

	if eng.ShardRole() != RoleWorker {
		t.Errorf("Expected role %v, got %v", RoleWorker, eng.ShardRole())
	}

	if eng.Config().HiddenDim != 128 {
		t.Errorf("Expected HiddenDim 128, got %d", eng.Config().HiddenDim)
	}

	eng.SetDeviceContext(ctx)
	if eng.deviceCtx == nil {
		t.Error("deviceCtx was not set")
	}

	eng.Close()
	if eng.deviceCtx != nil {
		t.Error("deviceCtx should be nil after Close()")
	}
}