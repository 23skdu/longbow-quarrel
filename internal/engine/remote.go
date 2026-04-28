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
	client *arrow_client.FlightClient
	config config.Config
	role   ShardRole
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

func (e *RemoteWorkerEngine) ForwardShardedLayer(ctx context.Context, layerIdx int, colStart, colEnd int, input *device.Tensor) (*device.Tensor, error) {
	if e.client == nil {
		return nil, fmt.Errorf("worker client not connected")
	}

	_ = colStart
	_ = colEnd
	_ = layerIdx
	_ = input
	_ = ctx

	return nil, nil
}
