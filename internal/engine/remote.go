package engine

import (
	"context"
	"fmt"
	"github.com/23skdu/longbow-quarrel/internal/arrow_client"
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

// RemoteWorkerEngine is an implementation of DistributedEngine that
// offloads execution to a remote Arrow Flight worker node.
type RemoteWorkerEngine struct {
	client    *arrow_client.FlightClient
	config    config.Config
	role      ShardRole
	deviceCtx *device.Context
}

func NewRemoteWorkerEngine(host string, port int, cfg config.Config) (*RemoteWorkerEngine, error) {
	client, err := arrow_client.NewFlightClient(host, port, host, port+1)
	if err != nil {
		return nil, err
	}

	return &RemoteWorkerEngine{
		client: client,
		config: cfg,
		role:   RoleWorker,
	}, nil
}

func (e *RemoteWorkerEngine) SetDeviceContext(ctx *device.Context) {
	e.deviceCtx = ctx
}

func (e *RemoteWorkerEngine) Close() {
	if e.deviceCtx != nil {
		e.deviceCtx.Free()
		e.deviceCtx = nil
	}
	if e.client != nil {
		_ = e.client.Close()
	}
}

func (e *RemoteWorkerEngine) ShardRole() ShardRole {
	return e.role
}

func (e *RemoteWorkerEngine) SyncWeights(ctx context.Context) error {
	// In a worker scenario, we might want to fetch weights from the master
	// or signaling that we are ready.
	return e.client.Connect(ctx)
}

func (e *RemoteWorkerEngine) ForwardShard(ctx context.Context, input *device.Tensor) (*device.Tensor, error) {
	// Push input tensor to worker, worker returns result shard
	// This uses the StreamEmbeddings or similar logic
	return nil, fmt.Errorf("ForwardShard not yet implemented over Flight")
}

func (e *RemoteWorkerEngine) Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error) {
	// Remote inference via Flight DoGet
	// The worker's Flight server handles the actual generation
	return nil, fmt.Errorf("remote Infer requires secondary Flight stream implementation")
}

func (e *RemoteWorkerEngine) InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(int)) (int, error) {
	return 0, fmt.Errorf("streaming remote inference not yet implemented")
}

func (e *RemoteWorkerEngine) ForwardBatch(batch *BatchDescriptor) ([]*device.Tensor, error) {
	return nil, fmt.Errorf("distributed ForwardBatch requires multi-node gRPC coordination")
}

func (e *RemoteWorkerEngine) Config() config.Config {
	return e.config
}

func (e *RemoteWorkerEngine) LoadAdapter(path, id string) error {
	return e.client.DoPut(context.Background(), nil, []string{id}, map[string]string{"path": path})
}

func (e *RemoteWorkerEngine) RollbackKV(seqID string, newPos int) error {
	return nil
}

func (e *RemoteWorkerEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	return nil, nil
}

// ForwardShardedLayer processes a layer shard via Arrow Flight RPC.
// This implements tensor parallelism where the master sends layer inputs to workers
// and receives computed outputs via Arrow Flight protocol.
func (e *RemoteWorkerEngine) ForwardShardedLayer(ctx context.Context, layerIdx int, colStart, colEnd int, input *device.Tensor) (*device.Tensor, error) {
	if e.client == nil {
		return nil, fmt.Errorf("worker client not connected")
	}

	if input == nil {
		return nil, fmt.Errorf("input tensor is nil")
	}

	// Serialize input tensor for RPC
	inputData := input.ToHostF32()
	if inputData == nil {
		return nil, fmt.Errorf("failed to get input data from device")
	}

	// Prepare metadata for layer computation
	meta := map[string]string{
		"layer_idx": fmt.Sprintf("%d", layerIdx),
		"col_start": fmt.Sprintf("%d", colStart),
		"col_end":   fmt.Sprintf("%d", colEnd),
		"rows":      fmt.Sprintf("%d", input.Rows()),
		"cols":      fmt.Sprintf("%d", input.Cols()),
	}

	// Send input tensor to worker via DoPut
	resultData, err := e.client.DoPutTensor(ctx, inputData, []int32{int32(input.Rows())}, meta) // #nosec G115 -- safe: Rows() is bounded by model config
	if err != nil {
		return nil, fmt.Errorf("DoPutTensor failed: %w", err)
	}

	if e.deviceCtx == nil {
		e.deviceCtx = device.NewContext()
	}

	rows := input.Rows()
	if rows <= 0 {
		rows = 1
	}
	cols := colEnd - colStart
	if cols <= 0 {
		cols = input.Cols()
	}
	if len(resultData) > 0 && rows > 0 {
		cols = len(resultData) / rows
	}

	outTensor := e.deviceCtx.NewTensorFP32(rows, cols)
	if len(resultData) > 0 {
		if err := outTensor.LoadFrom(resultData); err != nil {
			outTensor.Free()
			return nil, fmt.Errorf("failed to load shard result into tensor: %w", err)
		}
	}

	return outTensor, nil
}
