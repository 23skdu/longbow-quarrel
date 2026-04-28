package engine

import (
	"context"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

// DistributedEngine extends the base Engine with multi-node orchestration capabilities.
// It allows for Tensor Parallelism (TP) or Pipeline Parallelism (PP) by sharding
// the model logic across network-connected Arrow Flight nodes.
type DistributedEngine interface {
	Engine

	// ShardRole returns whether this node is a Master (coordinator) or Worker (sharded executor).
	ShardRole() ShardRole

	// SyncWeights synchronizes model shards across the cluster.
	SyncWeights(ctx context.Context) error

	// ForwardShard executes a forward pass on a specific shard of the model.
	// This is used for Tensor Parallelism where each node computes a partial hidden state.
	ForwardShard(ctx context.Context, input *device.Tensor) (*device.Tensor, error)

	// ForwardShardedLayer executes a forward pass on a specific layer shard.
	// Used for Tensor Parallelism where each node computes a partial hidden state for a layer.
	ForwardShardedLayer(ctx context.Context, layerIdx int, colStart, colEnd int, input *device.Tensor) (*device.Tensor, error)
}

// ShardRole defines the operational role of a node in a distributed cluster.
type ShardRole int

const (
	RoleStandalone ShardRole = iota
	RoleMaster
	RoleWorker
)
