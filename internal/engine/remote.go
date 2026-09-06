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
	if e.client == nil {
		return nil, fmt.Errorf("worker client not connected")
	}

	if input == nil {
		return nil, fmt.Errorf("input tensor is nil")
	}

	hiddenSize := e.config.Dim
	if hiddenSize <= 0 {
		hiddenSize = input.Cols()
	}

	current := input
	for layerIdx := 0; layerIdx < e.config.Layers; layerIdx++ {
		output, err := e.ForwardShardedLayer(ctx, layerIdx, 0, hiddenSize, current)
		if err != nil {
			return nil, fmt.Errorf("ForwardShardedLayer failed at layer %d: %w", layerIdx, err)
		}
		if layerIdx > 0 {
			current.Free()
		}
		current = output
	}

	return current, nil
}

func (e *RemoteWorkerEngine) Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error) {
	if e.client == nil {
		return nil, fmt.Errorf("worker client not connected")
	}

	if len(tokens) == 0 {
		return nil, fmt.Errorf("no input tokens provided")
	}

	ctx := context.Background()

	// Encode tokens as float32 vectors for Arrow Flight transport
	vectors := make([][]float32, len(tokens))
	ids := make([]string, len(tokens))
	for i, t := range tokens {
		vectors[i] = []float32{float32(t)}
		ids[i] = fmt.Sprintf("tok-%d", i)
	}

	// Build inference metadata for the worker
	meta := map[string]string{
		"op":           "infer",
		"count":        fmt.Sprintf("%d", count),
		"temperature":  fmt.Sprintf("%f", cfg.Temperature),
		"top_k":        fmt.Sprintf("%d", cfg.TopK),
		"top_p":        fmt.Sprintf("%f", cfg.TopP),
		"rep_penalty":  fmt.Sprintf("%f", cfg.RepPenalty),
		"presence_pen": fmt.Sprintf("%f", cfg.PresencePenalty),
		"freq_penalty": fmt.Sprintf("%f", cfg.FrequencyPenalty),
		"seed":         fmt.Sprintf("%d", cfg.Seed),
		"min_p":        fmt.Sprintf("%f", cfg.MinP),
	}

	// Send encoded tokens via Flight DoPut as a record batch
	if err := e.client.DoPut(ctx, vectors, ids, meta); err != nil {
		return nil, fmt.Errorf("DoPut failed for inference: %w", err)
	}

	// Retrieve generated token IDs via Flight DoGet
	result, err := e.client.DoGet(ctx, ids)
	if err != nil {
		return nil, fmt.Errorf("DoGet failed for inference results: %w", err)
	}

	// Convert result vectors back to token IDs
	generated := make([]int, 0, len(result.Vectors))
	for _, vec := range result.Vectors {
		if len(vec) > 0 {
			generated = append(generated, int(vec[0]))
		}
	}

	return generated, nil
}

func (e *RemoteWorkerEngine) InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(int)) ([]int, error) {
	if e.client == nil {
		return nil, fmt.Errorf("worker client not connected")
	}

	if len(tokens) == 0 {
		return nil, fmt.Errorf("no input tokens provided")
	}

	ctx := context.Background()

	// Encode tokens as float32 vectors for Arrow Flight transport
	vectors := make([][]float32, len(tokens))
	ids := make([]string, len(tokens))
	for i, t := range tokens {
		vectors[i] = []float32{float32(t)}
		ids[i] = fmt.Sprintf("tok-%d", i)
	}

	// Build inference metadata including streaming mode
	meta := map[string]string{
		"op":           "infer_stream",
		"count":        fmt.Sprintf("%d", count),
		"temperature":  fmt.Sprintf("%f", cfg.Temperature),
		"top_k":        fmt.Sprintf("%d", cfg.TopK),
		"top_p":        fmt.Sprintf("%f", cfg.TopP),
		"rep_penalty":  fmt.Sprintf("%f", cfg.RepPenalty),
		"presence_pen": fmt.Sprintf("%f", cfg.PresencePenalty),
		"freq_penalty": fmt.Sprintf("%f", cfg.FrequencyPenalty),
		"seed":         fmt.Sprintf("%d", cfg.Seed),
		"min_p":        fmt.Sprintf("%f", cfg.MinP),
		"stream":       "true",
	}

	// Send encoded tokens via Flight DoPut as a record batch
	if err := e.client.DoPut(ctx, vectors, ids, meta); err != nil {
		return nil, fmt.Errorf("DoPut failed for streaming inference: %w", err)
	}

	// Retrieve generated token IDs via Flight DoGet
	result, err := e.client.DoGet(ctx, ids)
	if err != nil {
		return nil, fmt.Errorf("DoGet failed for streaming inference results: %w", err)
	}

	// Convert result vectors to token IDs and invoke the callback for each
	generated := make([]int, 0, len(result.Vectors))
	for _, vec := range result.Vectors {
		if len(vec) > 0 {
			tok := int(vec[0])
			generated = append(generated, tok)
			if callback != nil {
				callback(tok)
			}
		}
	}

	return generated, nil
}

func (e *RemoteWorkerEngine) ForwardBatch(batch *BatchDescriptor) ([]*device.Tensor, error) {
	if batch == nil || len(batch.Sequences) == 0 {
		return nil, nil
	}

	batchSize := len(batch.Sequences)
	results := make([]*device.Tensor, batchSize)
	ctx := context.Background()

	hiddenSize := e.config.Dim
	if hiddenSize <= 0 {
		hiddenSize = 576
	}

	for idx := range batch.Sequences {
		seq := batch.Sequences[idx]
		if seq == nil {
			for j := 0; j < idx; j++ {
				if results[j] != nil {
					results[j].Free()
				}
			}
			return nil, fmt.Errorf("nil sequence at index %d", idx)
		}

		start := batch.Offsets[idx]
		var end int
		if idx < batchSize-1 {
			end = batch.Offsets[idx+1]
		} else {
			end = len(batch.Tokens)
		}

		seqTokens := batch.Tokens[start:end]
		if len(seqTokens) == 0 {
			for j := 0; j < idx; j++ {
				if results[j] != nil {
					results[j].Free()
				}
			}
			return nil, fmt.Errorf("empty tokens for sequence %d", idx)
		}

		// Encode tokens as float32 for Flight transport
		tokenData := make([]float32, len(seqTokens))
		for i, t := range seqTokens {
			tokenData[i] = float32(t)
		}

		meta := map[string]string{
			"op":          "forward_batch",
			"seq_idx":     fmt.Sprintf("%d", idx),
			"context_len": fmt.Sprintf("%d", batch.ContextLens[idx]),
			"hidden_size": fmt.Sprintf("%d", hiddenSize),
			"token_count": fmt.Sprintf("%d", len(seqTokens)),
		}
		if idx < len(batch.IsDecode) {
			meta["is_decode"] = fmt.Sprintf("%v", batch.IsDecode[idx])
		}
		if idx < len(batch.AdapterIDs) && batch.AdapterIDs[idx] != "" {
			meta["adapter_id"] = batch.AdapterIDs[idx]
		}

		// Forward each sequence through the remote worker via DoPutTensor
		// #nosec G115 -- safe: sequence token count is bounded and fits in int32
		resultData, err := e.client.DoPutTensor(ctx, tokenData, []int32{int32(len(seqTokens))}, meta)
		if err != nil {
			for j := 0; j < idx; j++ {
				if results[j] != nil {
					results[j].Free()
				}
			}
			return nil, fmt.Errorf("ForwardBatch DoPutTensor failed for sequence %d: %w", idx, err)
		}

		if e.deviceCtx == nil {
			e.deviceCtx = device.NewContext()
		}

		cols := hiddenSize
		if len(resultData) > 0 && len(resultData) <= hiddenSize {
			cols = len(resultData)
		}

		tensor := e.deviceCtx.NewTensorFP32(1, cols)
		if len(resultData) > 0 {
			if err := tensor.LoadFrom(resultData); err != nil {
				tensor.Free()
				for j := 0; j < idx; j++ {
					if results[j] != nil {
						results[j].Free()
					}
				}
				return nil, fmt.Errorf("failed to load result for sequence %d: %w", idx, err)
			}
		}
		results[idx] = tensor
	}

	return results, nil
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
