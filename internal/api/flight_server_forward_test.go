package api

import (
	"context"
	"net"
	"strconv"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/arrow_client"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/flight"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// startTestFlightServer starts the inference Flight server on a free port and
// returns its host and port.
func startTestFlightServer(t *testing.T) (string, int) {
	t.Helper()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}

	server := NewInferenceFlightServer(lis.Addr().String(), &engine.MockEngine{}, &tokenizer.Tokenizer{})

	go func() { _ = server.ServeListener(lis) }()

	time.Sleep(200 * time.Millisecond)

	host, portStr, err := net.SplitHostPort(lis.Addr().String())
	if err != nil {
		t.Fatalf("failed to split host port: %v", err)
	}
	port, _ := strconv.Atoi(portStr)
	return host, port
}

// TestFlightServer_TensorForwardIsRejected checks that the worker refuses the
// tensor forward operation instead of silently discarding the payload.
//
// The server has no engine API that runs a hidden state through a layer range
// and returns the output hidden state. It used to discard the payload and
// answer with an empty PutResult, which the client read as "the worker
// processed it but sent no data" and turned into a **zero-filled** hidden
// state: the master then continued with zero activations and produced
// garbage, with no error anywhere.
func TestFlightServer_TensorForwardIsRejected(t *testing.T) {
	host, port := startTestFlightServer(t)

	client, err := arrow_client.NewFlightClient(host, port, host, port)
	if err != nil {
		t.Fatalf("failed to create the client: %v", err)
	}
	if err := client.Connect(context.Background()); err != nil {
		t.Fatalf("failed to connect: %v", err)
	}
	defer func() { _ = client.Close() }()

	// A tensor forward against a worker that does not implement it must
	// fail, not return an empty result.
	data := []float32{1, 2, 3, 4}
	result, err := client.DoPutTensor(context.Background(), data, []int32{1}, map[string]string{"op": "forward_batch"})
	if err == nil {
		t.Fatalf("expected an error, got a result of %v", result)
	}
	if len(result) != 0 {
		t.Errorf("expected no result data, got %v", result)
	}
}

// TestFlightServer_ControlMessagesAreAccepted checks that the fire-and-forget
// operations still work.
//
// The control messages (load an adapter, and so on) share the DoPut
// handler with the tensor forward, so rejecting the forward must not reject
// them.
func TestFlightServer_ControlMessagesAreAccepted(t *testing.T) {
	host, port := startTestFlightServer(t)

	client, err := arrow_client.NewFlightClient(host, port, host, port)
	if err != nil {
		t.Fatalf("failed to create the client: %v", err)
	}
	if err := client.Connect(context.Background()); err != nil {
		t.Fatalf("failed to connect: %v", err)
	}
	defer func() { _ = client.Close() }()

	// A control message carries no tensor and must be accepted. It uses the
	// "embeddings" descriptor path, which is distinct from the tensor
	// forward's "compute/layer".
	if err := client.DoPut(context.Background(), [][]float32{{1, 2, 3, 4}}, []string{"adapter-id"},
		map[string]string{"path": "/tmp/adapter.gguf"}); err != nil {
		t.Errorf("expected the control message to be accepted, got: %v", err)
	}
}

// TestFlightServer_TensorForwardReturnsUnimplemented checks the status code
// the worker returns for the tensor forward.
func TestFlightServer_TensorForwardReturnsUnimplemented(t *testing.T) {
	host, port := startTestFlightServer(t)

	client, err := flight.NewClientWithMiddleware(host+":"+strconv.Itoa(port), nil, nil,
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("failed to create the flight client: %v", err)
	}
	defer func() { _ = client.Close() }()

	stream, err := client.DoPut(context.Background())
	if err != nil {
		t.Fatalf("failed to create the DoPut stream: %v", err)
	}

	allocator := memory.NewGoAllocator()
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "row_idx", Type: arrow.PrimitiveTypes.Int32},
		{Name: "data", Type: arrow.FixedSizeListOf(4, arrow.PrimitiveTypes.Float32)},
	}, nil)

	builder := array.NewRecordBuilder(allocator, schema)
	defer builder.Release()
	builder.Field(0).(*array.Int32Builder).Append(1)
	list := builder.Field(1).(*array.FixedSizeListBuilder)
	list.Append(true)
	values := list.ValueBuilder().(*array.Float32Builder)
	values.Append(1)
	values.Append(2)
	values.Append(3)
	values.Append(4)

	rec := builder.NewRecordBatch()
	defer rec.Release()

	writer := flight.NewRecordWriter(stream, ipc.WithSchema(schema))
	writer.SetFlightDescriptor(&flight.FlightDescriptor{
		Type: flight.DescriptorPATH,
		Path: []string{"compute", "layer"},
	})
	if err := writer.Write(rec); err != nil {
		t.Fatalf("failed to write the record: %v", err)
	}
	_ = writer.Close()
	_ = stream.CloseSend()

	_, err = stream.Recv()
	if err == nil {
		t.Fatal("expected the worker to reject the tensor forward operation")
	}
	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected a gRPC status error, got %v", err)
	}
	if st.Code() != codes.Unimplemented {
		t.Errorf("expected the Unimplemented code, got %v (%s)", st.Code(), st.Message())
	}
}
