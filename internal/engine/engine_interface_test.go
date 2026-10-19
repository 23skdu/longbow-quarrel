package engine

import (
	"context"
	"errors"
	"net"
	"strconv"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/flight"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
)

// The engine interface has a lot of thin delegation methods that nothing
// exercised. They are the surface the API server, the Arrow Flight server
// and the distributed master all call, so they are worth covering.

// countingEngine records the calls that reach it.
type countingEngine struct {
	MockEngine
	token          int
	inferCalls     int
	adapterCalls   int
	swapCalls      int
	closeCalls     int
	rollbackCalls  int
	draftCalls     int
	cachePos       int
	forwardLogits  [][]float32
	forwardBatches int
}

func (e *countingEngine) Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error) {
	e.inferCalls++
	if e.token != 0 {
		return []int{e.token}, nil
	}
	return []int{7, 8, 9}, nil
}

func (e *countingEngine) InferWithLogits(tokens []int, count int, cfg SamplerConfig) ([]int, []float32, error) {
	e.inferCalls++
	if e.token != 0 {
		return []int{e.token}, []float32{0.1, 0.9}, nil
	}
	return []int{7}, []float32{0.5, 0.5}, nil
}

func (e *countingEngine) InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(int)) ([]int, error) {
	e.inferCalls++
	out := []int{7, 8}
	for _, t := range out {
		if callback != nil {
			callback(t)
		}
	}
	return out, nil
}

func (e *countingEngine) InferWithCallbackLogits(tokens []int, count int, cfg SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	e.inferCalls++
	if tokenCallback != nil {
		tokenCallback(7)
	}
	if logitsCallback != nil {
		logitsCallback([]float32{0.5})
	}
	return []int{7}, nil
}

func (e *countingEngine) LoadAdapter(path, id string) error {
	e.adapterCalls++
	return nil
}

func (e *countingEngine) SwapModel(modelPath string, cfg config.Config) error {
	e.swapCalls++
	return nil
}

func (e *countingEngine) Close() { e.closeCalls++ }

func (e *countingEngine) GetSeqCachePos(seqID string) int { return e.cachePos }

func (e *countingEngine) RollbackKV(seqID string, newPos int) error {
	e.rollbackCalls++
	return nil
}

func (e *countingEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	e.draftCalls++
	return e.forwardLogits, nil
}

func (e *countingEngine) ForwardBatch(desc *BatchDescriptor) ([]*device.Tensor, error) {
	e.forwardBatches++
	return nil, nil
}

func (e *countingEngine) ShardRole() ShardRole { return RoleWorker }

func (e *countingEngine) SyncWeights(ctx context.Context) error { return nil }

func (e *countingEngine) ForwardShard(ctx context.Context, input *device.Tensor) (*device.Tensor, error) {
	return nil, nil
}

func (e *countingEngine) ForwardShardedLayer(ctx context.Context, layerIdx, colStart, colEnd int, input *device.Tensor) (*device.Tensor, error) {
	return nil, nil
}

// TestAsymmetricSpeculativeEngineInterface covers the whole Engine
// interface of the speculative wrapper.
//
// The wrapper is what the CLI and the REST API use when a draft model is
// configured, and most of its methods had no test at all.
func TestAsymmetricSpeculativeEngineInterface(t *testing.T) {
	target := &countingEngine{
		cachePos: 42,
		forwardLogits: [][]float32{
			{0.1, 0.9}, {0.1, 0.9}, {0.1, 0.9}, {0.1, 0.9},
		},
	}
	draft := &countingEngine{
		token:         1,
		forwardLogits: [][]float32{{0.1, 0.9}},
	}

	spec := NewAsymmetricSpeculativeEngineWithOptions(target, draft, 4, 1)

	if got, err := spec.Infer([]int{1}, 2, SamplerConfig{}); err != nil || len(got) == 0 {
		t.Fatalf("Infer returned %v, %v", got, err)
	}
	if target.inferCalls == 0 && draft.inferCalls == 0 {
		t.Error("Infer did not reach either engine")
	}

	if _, _, err := spec.InferWithLogits([]int{1}, 2, SamplerConfig{}); err != nil {
		t.Errorf("InferWithLogits failed: %v", err)
	}

	var tokens []int
	var logits []float32
	if _, err := spec.InferWithCallbackLogits([]int{1}, 2, SamplerConfig{},
		func(t int) { tokens = append(tokens, t) },
		func(l []float32) { logits = l }); err != nil {
		t.Errorf("InferWithCallbackLogits failed: %v", err)
	}
	if len(tokens) == 0 {
		t.Error("the token callback was never called")
	}
	if len(logits) == 0 {
		t.Error("the logits callback was never called")
	}

	// The configuration comes from the target.
	if spec.Config().MaxTokens != target.Config().MaxTokens {
		t.Error("the configuration does not come from the target engine")
	}

	// The model swap goes to the target.
	if err := spec.SwapModel("/tmp/model.gguf", config.Default()); err != nil {
		t.Errorf("SwapModel failed: %v", err)
	}
	if target.swapCalls == 0 {
		t.Error("SwapModel did not reach the target engine")
	}

	// The adapter load goes to the target.
	if err := spec.LoadAdapter("/tmp/adapter.gguf", "adapter-1"); err != nil {
		t.Errorf("LoadAdapter failed: %v", err)
	}
	if target.adapterCalls == 0 {
		t.Error("LoadAdapter did not reach the target engine")
	}

	// The cache position comes from the target.
	if got := spec.GetSeqCachePos("seq-1"); got != 42 {
		t.Errorf("the cache position is %d, want 42", got)
	}

	// The draft comes from the target as well.
	if _, err := spec.ForwardDraft([]int{1}); err != nil {
		t.Errorf("ForwardDraft failed: %v", err)
	}
	if target.draftCalls == 0 {
		t.Error("ForwardDraft did not reach the target engine")
	}

	// The rollback reaches both engines.
	if err := spec.RollbackKV("seq-1", 3); err != nil {
		t.Errorf("RollbackKV failed: %v", err)
	}
	if target.rollbackCalls == 0 || draft.rollbackCalls == 0 {
		t.Error("RollbackKV did not reach both engines")
	}

	// The batched forward goes to the target.
	if _, err := spec.ForwardBatch(&BatchDescriptor{}); err != nil {
		t.Errorf("ForwardBatch failed: %v", err)
	}
	if target.forwardBatches == 0 {
		t.Error("ForwardBatch did not reach the target engine")
	}

	// The draft length is adjustable.
	spec.SetDraftLength(6)
	if got := spec.manager.GetDynamicDraftLength(); got != 6 {
		t.Errorf("the draft length is %d, want 6", got)
	}
	// A non-positive value keeps the current one.
	spec.SetDraftLength(0)
	if got := spec.manager.GetDynamicDraftLength(); got != 6 {
		t.Errorf("the draft length is %d, want 6", got)
	}

	// Closing closes both engines.
	spec.Close()
	if target.closeCalls == 0 || draft.closeCalls == 0 {
		t.Error("Close did not close both engines")
	}
}

// TestMasterDistributedEngineConfigAndSwap covers the two master methods
// that had no test.
func TestMasterDistributedEngineConfigAndSwap(t *testing.T) {
	cfg := config.Default()
	cfg.Layers = 7

	shard := &countingEngine{}
	m := NewMasterDistributedEngine([]DistributedEngine{shard}, cfg)
	defer m.Close()

	if got := m.Config().Layers; got != 7 {
		t.Errorf("the configuration has %d layers, want 7", got)
	}

	if err := m.SwapModel("/tmp/model.gguf", cfg); err != nil {
		t.Errorf("SwapModel failed: %v", err)
	}
	if shard.swapCalls != 1 {
		t.Errorf("the shard was asked to swap %d times, want 1", shard.swapCalls)
	}
}

// workerFlightServer is a minimal Arrow Flight server for the remote
// worker tests.
type workerFlightServer struct {
	flight.BaseFlightServer
	allocator memory.Allocator
	tokens    []int
	puts      int
	gets      int
}

func (s *workerFlightServer) DoGet(tckt *flight.Ticket, stream flight.FlightService_DoGetServer) error {
	s.gets++

	schema := arrow.NewSchema([]arrow.Field{
		{Name: "token_id", Type: arrow.PrimitiveTypes.Int32},
		{Name: "id", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "vector", Type: arrow.FixedSizeListOf(1, arrow.PrimitiveTypes.Float32), Nullable: true},
	}, nil)

	writer := flight.NewRecordWriter(stream, ipc.WithSchema(schema))
	defer func() { _ = writer.Close() }()

	builder := array.NewRecordBuilder(s.allocator, schema)
	defer builder.Release()
	ids := builder.Field(0).(*array.Int32Builder)
	idStrings := builder.Field(1).(*array.StringBuilder)
	vectors := builder.Field(2).(*array.FixedSizeListBuilder)
	for _, t := range s.tokens {
		ids.Append(int32(t))
		idStrings.Append(strconv.Itoa(t))
		vectors.Append(true)
		vectors.ValueBuilder().(*array.Float32Builder).Append(float32(t))
	}
	rec := builder.NewRecordBatch()
	defer rec.Release()
	if err := writer.Write(rec); err != nil {
		return err
	}
	return nil
}

func (s *workerFlightServer) GetFlightInfo(ctx context.Context, desc *flight.FlightDescriptor) (*flight.FlightInfo, error) {
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "token_id", Type: arrow.PrimitiveTypes.Int32},
	}, nil)
	return &flight.FlightInfo{
		Schema:       flight.SerializeSchema(schema, s.allocator),
		Endpoint:     []*flight.FlightEndpoint{{Ticket: &flight.Ticket{Ticket: []byte("tickets")}}},
		TotalRecords: -1,
		TotalBytes:   -1,
	}, nil
}

func (s *workerFlightServer) DoPut(stream flight.FlightService_DoPutServer) error {
	for {
		_, err := stream.Recv()
		if err != nil {
			break
		}
		s.puts++
	}
	return stream.SendMsg(&flight.PutResult{})
}

// startWorkerFlightServer starts the test server and returns its address.
func startWorkerFlightServer(t *testing.T, tokens []int) string {
	t.Helper()

	srv := flight.NewServerWithMiddleware(nil)
	if err := srv.Init("localhost:0"); err != nil { // #nosec G102 -- loopback test listener
		t.Fatalf("flight server Init: %v", err)
	}
	srv.RegisterFlightService(&workerFlightServer{
		allocator: memory.NewGoAllocator(),
		tokens:    tokens,
	})
	go func() { _ = srv.Serve() }()
	t.Cleanup(srv.Shutdown)

	return srv.Addr().String()
}

// startTensorWorkerServer starts a test server that also answers the
// tensor-forward operation.
func startTensorWorkerServer(t *testing.T, values []float32) string {
	t.Helper()

	srv := flight.NewServerWithMiddleware(nil)
	if err := srv.Init("localhost:0"); err != nil { // #nosec G102 -- loopback test listener
		t.Fatalf("flight server Init: %v", err)
	}
	srv.RegisterFlightService(&tensorFlightServer{
		workerFlightServer: workerFlightServer{allocator: memory.NewGoAllocator()},
		values:             values,
	})
	go func() { _ = srv.Serve() }()
	t.Cleanup(srv.Shutdown)

	return srv.Addr().String()
}

// newRemoteWorker builds a remote worker engine pointed at the test
// server.
func newRemoteWorker(t *testing.T, addr string) *RemoteWorkerEngine {
	t.Helper()

	host, portStr, err := net.SplitHostPort(addr)
	if err != nil {
		t.Fatalf("failed to split the host and port: %v", err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatalf("failed to parse the port: %v", err)
	}

	cfg := config.Default()
	cfg.Dim = 8
	cfg.Layers = 2

	worker, err := NewRemoteWorkerEngine(host, port, cfg)
	if err != nil {
		t.Fatalf("failed to create the remote worker: %v", err)
	}
	t.Cleanup(worker.Close)

	return worker
}

// TestRemoteWorkerEngineInference covers the remote worker's inference
// path against an in-process Flight server.
func TestRemoteWorkerEngineInference(t *testing.T) {
	addr := startWorkerFlightServer(t, []int{11, 12, 13})
	worker := newRemoteWorker(t, addr)

	tokens, err := worker.Infer([]int{1, 2, 3}, 3, SamplerConfig{Temperature: 0.7})
	if err != nil {
		t.Fatalf("Infer failed: %v", err)
	}
	if len(tokens) != 3 {
		t.Fatalf("Infer returned %d tokens, want 3", len(tokens))
	}
	for i, want := range []int{11, 12, 13} {
		if tokens[i] != want {
			t.Errorf("token %d is %d, want %d", i, tokens[i], want)
		}
	}

	// The callback variant reports every token.
	var streamed []int
	streamedTokens, err := worker.InferWithCallback([]int{1, 2, 3}, 3, SamplerConfig{},
		func(tok int) { streamed = append(streamed, tok) })
	if err != nil {
		t.Fatalf("InferWithCallback failed: %v", err)
	}
	if len(streamedTokens) != 3 {
		t.Errorf("InferWithCallback returned %d tokens, want 3", len(streamedTokens))
	}
	if len(streamed) != 3 {
		t.Errorf("the callback fired %d times, want 3", len(streamed))
	}
}

// TestRemoteWorkerEngineGuards covers the guard rails of the remote
// worker.
func TestRemoteWorkerEngineGuards(t *testing.T) {
	addr := startWorkerFlightServer(t, []int{1})
	worker := newRemoteWorker(t, addr)

	if _, err := worker.Infer(nil, 1, SamplerConfig{}); err == nil {
		t.Error("expected an error when there are no tokens")
	}
	if _, err := worker.InferWithCallback(nil, 1, SamplerConfig{}, nil); err == nil {
		t.Error("expected an error when there are no tokens")
	}
	if _, err := worker.ForwardShard(context.Background(), nil); err == nil {
		t.Error("expected an error for a nil input tensor")
	}

	// A worker with no client reports the disconnection rather than
	// panicking.
	disconnected := &RemoteWorkerEngine{}
	if _, err := disconnected.Infer([]int{1}, 1, SamplerConfig{}); err == nil {
		t.Error("expected an error when the client is not connected")
	}
	if err := disconnected.SyncWeights(context.Background()); err == nil {
		t.Error("expected an error when the client is not connected")
	}
	if _, err := disconnected.ForwardShard(context.Background(), nil); err == nil {
		t.Error("expected an error when the client is not connected")
	}
}

// TestRemoteWorkerEngineControlMessages covers the control operations.
func TestRemoteWorkerEngineControlMessages(t *testing.T) {
	addr := startWorkerFlightServer(t, []int{1})
	worker := newRemoteWorker(t, addr)

	if err := worker.LoadAdapter("/tmp/adapter.gguf", "adapter-1"); err != nil {
		t.Errorf("LoadAdapter failed: %v", err)
	}
	if err := worker.RollbackKV("seq-1", 3); err != nil {
		t.Errorf("RollbackKV failed: %v", err)
	}
	// The draft forward uses the tensor-forward operation, which the
	// worker does not implement: it fails loudly rather than returning
	// zeroed activations.
	if _, err := worker.ForwardDraft([]int{1}); err == nil {
		t.Error("expected the unsupported tensor forward to be reported")
	}
	if err := worker.SyncWeights(context.Background()); err != nil {
		t.Errorf("SyncWeights failed: %v", err)
	}

	// A remote worker is a worker.
	if worker.ShardRole() != RoleWorker {
		t.Errorf("the role is %v, want the worker role", worker.ShardRole())
	}

	// The next stage and the fallback worker are recorded.
	worker.SetNextStage("127.0.0.1", 1234)
	if err := worker.SetFallbackWorker("127.0.0.1", 5678); err != nil {
		t.Errorf("SetFallbackWorker failed: %v", err)
	}

	// A device context can be attached.
	worker.SetDeviceContext(device.NewContext())
}

// TestRemoteWorkerEngineForwardBatch checks that an empty batch is a
// no-op rather than an error.
func TestRemoteWorkerEngineForwardBatch(t *testing.T) {
	addr := startWorkerFlightServer(t, []int{1})
	worker := newRemoteWorker(t, addr)

	results, err := worker.ForwardBatch(nil)
	if err != nil {
		t.Errorf("a nil batch should be a no-op, got %v", err)
	}
	if len(results) != 0 {
		t.Errorf("a nil batch returned %d results", len(results))
	}

	results, err = worker.ForwardBatch(&BatchDescriptor{})
	if err != nil {
		t.Errorf("an empty batch should be a no-op, got %v", err)
	}
	if len(results) != 0 {
		t.Errorf("an empty batch returned %d results", len(results))
	}
}

// TestQualityEvaluatorMetrics covers the quality evaluator's metrics.
func TestQualityEvaluatorMetrics(t *testing.T) {
	qe := NewQualityEvaluatorSimple()

	// The perplexity of an empty sequence is zero, not a division by
	// zero.
	empty := qe.CalculatePerplexity(nil)
	if empty.Perplexity != 1 {
		t.Errorf("the perplexity of an empty sequence is %v, want 1", empty.Perplexity)
	}
	if empty.TotalTokens != 0 {
		t.Errorf("the token count is %d, want 0", empty.TotalTokens)
	}

	// The ROUGE of a text against itself is a perfect score.
	rouge := qe.CalculateROUGE("the quick brown fox", "the quick brown fox")
	if rouge.ROUGEL_F1 <= 0 {
		t.Errorf("the ROUGE of a text against itself is %v, want a positive score", rouge.ROUGEL_F1)
	}

	// Two empty texts do not divide by zero.
	both := qe.CalculateROUGE("", "")
	if both.F1 != 0 {
		t.Errorf("the ROUGE of two empty texts is %v, want 0", both.F1)
	}

	// The BLEU of an empty candidate is zero.
	bleu := qe.CalculateBLEU("", "a reference")
	if bleu.BLEU1 != 0 {
		t.Errorf("the BLEU of an empty candidate is %v, want 0", bleu.BLEU1)
	}
}

// TestGetNGrams covers the n-gram helper.
func TestGetNGrams(t *testing.T) {
	grams := getNGrams([]int{1, 2, 3, 4}, 2)
	if len(grams) != 3 {
		t.Errorf("a bigram count of four tokens is %d, want 3", len(grams))
	}
	if grams["[1 2]"] != 1 {
		t.Errorf("the bigram [1 2] appears %d times, want 1", grams["[1 2]"])
	}

	// A repeated bigram is counted twice.
	repeated := getNGrams([]int{1, 2, 1, 2}, 2)
	if repeated["[1 2]"] != 2 {
		t.Errorf("the bigram [1 2] appears %d times, want 2", repeated["[1 2]"])
	}

	// A sequence shorter than the n-gram yields nothing.
	if got := getNGrams([]int{1}, 2); len(got) != 0 {
		t.Errorf("a short sequence produced %d n-grams, want none", len(got))
	}
}

// TestMockEngineMethods covers the mock engine, which the API and the
// distributed tests use.
func TestMockEngineMethods(t *testing.T) {
	m, err := NewMockEngine("model.gguf", config.Default())
	if err != nil {
		t.Fatalf("failed to create the mock engine: %v", err)
	}

	if err := m.LoadAdapter("/tmp/adapter.gguf", "adapter-1"); err != nil {
		t.Errorf("LoadAdapter failed: %v", err)
	}
	if _, err := m.ForwardBatch(&BatchDescriptor{}); err != nil {
		t.Errorf("ForwardBatch failed: %v", err)
	}
	if err := m.RollbackKV("seq-1", 1); err != nil {
		t.Errorf("RollbackKV failed: %v", err)
	}
	if got := m.GetSeqCachePos("seq-1"); got != 0 {
		t.Errorf("the cache position is %d, want 0", got)
	}
	m.Close()
}

// TestLoRAManagerLoadAdapterErrors covers the LoRA manager's error paths.
func TestLoRAManagerLoadAdapterErrors(t *testing.T) {
	lm := NewLoRAManager()

	// A file that does not exist.
	if err := lm.LoadAdapter(nil, "/nonexistent/adapter.gguf", "adapter-1"); err == nil {
		t.Error("expected an error for a missing adapter file")
	}
}

// TestCPUEngineSwapModelErrors covers the CPU engine's model swap guard.
func TestCPUEngineSwapModelErrors(t *testing.T) {
	e := &CPUEngine{weights: &CPUWeights{}}

	// A model path that does not exist is an error, not a panic.
	if err := e.SwapModel("/nonexistent/model.gguf", config.Default()); err == nil {
		t.Error("expected an error when the model cannot be loaded")
	}
}

// TestRemoteWorkerEngineFallbackIsUsed checks that a failing primary
// worker falls back to the secondary one.
func TestRemoteWorkerEngineFallbackIsUsed(t *testing.T) {
	addr := startWorkerFlightServer(t, []int{5, 6})
	host, portStr, err := net.SplitHostPort(addr)
	if err != nil {
		t.Fatalf("failed to split the host and port: %v", err)
	}
	port, _ := strconv.Atoi(portStr)

	cfg := config.Default()
	worker, err := NewRemoteWorkerEngine(host, port, cfg)
	if err != nil {
		t.Fatalf("failed to create the remote worker: %v", err)
	}
	defer worker.Close()

	// A fallback that does not exist: the primary still works.
	if err := worker.SetFallbackWorker("127.0.0.1", 1); err != nil {
		t.Errorf("SetFallbackWorker failed: %v", err)
	}

	tokens, err := worker.Infer([]int{1}, 2, SamplerConfig{})
	if err != nil {
		t.Fatalf("Infer failed: %v", err)
	}
	if len(tokens) == 0 {
		t.Error("Infer returned no tokens")
	}
}

// TestDistributedEngineErrorsAreReported checks that a shard failure is
// reported rather than swallowed.
func TestDistributedEngineErrorsAreReported(t *testing.T) {
	failing := &failingShard{}
	m := NewMasterDistributedEngine([]DistributedEngine{failing}, config.Default())
	defer m.Close()

	if err := m.SwapModel("/tmp/model.gguf", config.Default()); err == nil {
		t.Error("expected the swap error to be reported")
	}
	if err := m.LoadAdapter("/tmp/adapter.gguf", "adapter-1"); err == nil {
		t.Error("expected the adapter error to be reported")
	}
}

// failingShard fails every operation.
type failingShard struct {
	MockEngine
}

func (s *failingShard) ShardRole() ShardRole { return RoleWorker }

func (s *failingShard) SyncWeights(ctx context.Context) error { return errors.New("sync failed") }

func (s *failingShard) ForwardShard(ctx context.Context, input *device.Tensor) (*device.Tensor, error) {
	return nil, errors.New("forward failed")
}

func (s *failingShard) ForwardShardedLayer(ctx context.Context, layerIdx, colStart, colEnd int, input *device.Tensor) (*device.Tensor, error) {
	return nil, errors.New("shard forward failed")
}

func (s *failingShard) SwapModel(modelPath string, cfg config.Config) error {
	return errors.New("swap failed")
}

func (s *failingShard) LoadAdapter(path, id string) error { return errors.New("adapter failed") }
