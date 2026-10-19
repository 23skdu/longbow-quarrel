package api

import (
	"context"
	"fmt"
	"io"
	"net"
	"strconv"

	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/flight"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// InferenceFlightServer handles bidirectional Arrow streaming for the generation endpoints.
type InferenceFlightServer struct {
	flight.BaseFlightServer
	allocator memory.Allocator
	addr      string
	engine    engine.Engine
	tokenizer *tokenizer.Tokenizer
}

// NewInferenceFlightServer initializes the Arrow Flight generation server.
func NewInferenceFlightServer(addr string, e engine.Engine, t *tokenizer.Tokenizer) *InferenceFlightServer {
	return &InferenceFlightServer{
		allocator: memory.NewGoAllocator(),
		addr:      addr,
		engine:    e,
		tokenizer: t,
	}
}

// DoGet streams generated tokens directly using Arrow Array formatting.
func (s *InferenceFlightServer) DoGet(tckt *flight.Ticket, stream flight.FlightService_DoGetServer) error {
	prompt := string(tckt.Ticket)
	if prompt == "" {
		prompt = "Continue training" // Default fallback
	}

	promptTokens := s.tokenizer.Encode(prompt)

	// Metadata for GPU discovery
	meta := arrow.NewMetadata([]string{"QUARREL:device_id"}, []string{"0"})

	// The schema carries the token id three ways: as an integer, as the
	// decoded text, and in the "id" and "vector" columns the Arrow Flight
	// client reads.
	//
	// The client looks for "id" and "vector" and ignores everything else,
	// so a server that only sent "token_id", "logits" and "text" produced
	// a stream the client could not read: the remote worker came back
	// with no tokens at all, and with no error.
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "token_id", Type: arrow.PrimitiveTypes.Int32},
		{Name: "logits", Type: arrow.FixedSizeListOf(int32(10), arrow.PrimitiveTypes.Float32), Nullable: true},
		{Name: "text", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "id", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "vector", Type: arrow.FixedSizeListOf(1, arrow.PrimitiveTypes.Float32), Nullable: true},
	}, &meta)

	writer := flight.NewRecordWriter(stream, ipc.WithSchema(schema))
	defer func() { _ = writer.Close() }()

	// Capture generated tokens into this channel
	tokenChan := make(chan int, 128)
	errChan := make(chan error, 1)

	go func() {
		_, err := s.engine.InferWithCallback(promptTokens, 100, engine.SamplerConfig{Temperature: 0.7}, func(token int) {
			tokenChan <- token
		})
		close(tokenChan)
		errChan <- err
	}()

	for token := range tokenChan {
		builder := array.NewRecordBuilder(s.allocator, schema)

		tokenIDBuilder := builder.Field(0).(*array.Int32Builder)
		tokenIDBuilder.Append(int32(token)) // #nosec G115 -- safe: token IDs are bounded by vocab size

		// Ensure all fields have same row count
		builder.Field(1).(*array.FixedSizeListBuilder).AppendNull()

		text := s.tokenizer.Decode([]int{token})
		textBuilder := builder.Field(2).(*array.StringBuilder)
		textBuilder.Append(text)

		// The columns the Arrow Flight client reads.
		idBuilder := builder.Field(3).(*array.StringBuilder)
		idBuilder.Append(strconv.Itoa(token))

		vectorBuilder := builder.Field(4).(*array.FixedSizeListBuilder)
		vectorBuilder.Append(true)
		vectorBuilder.ValueBuilder().(*array.Float32Builder).Append(float32(token))

		rec := builder.NewRecordBatch()
		err := writer.Write(rec)
		rec.Release()
		builder.Release()

		if err != nil {
			return err
		}
	}

	return <-errChan
}

// DoPut receives an Arrow stream and stores it as a sequence.
//
// The Flight descriptor distinguishes the two kinds of request:
//
//	["compute", "layer"]  a tensor forward, which needs a result tensor back
//	anything else         a fire-and-forget control message
//
// The tensor forward is not implemented on the server side: there is no
// engine API that runs a hidden state through a layer range and returns the
// output hidden state. The previous implementation discarded the payload and
// answered with an empty PutResult, which the client interpreted as "the
// worker processed it but sent no data" and turned into a **zero-filled**
// hidden state. Failing loudly is much better than silently corrupting the
// activations.
func (s *InferenceFlightServer) DoPut(stream flight.FlightService_DoPutServer) error {
	first := true
	for {
		fd, err := stream.Recv()
		if err == io.EOF {
			return stream.SendMsg(&flight.PutResult{})
		}
		if err != nil {
			return err
		}
		if first {
			first = false
			// The descriptor is carried on the first message of the stream.
			if desc := fd.GetFlightDescriptor(); desc != nil && len(desc.Path) >= 2 &&
				desc.Path[0] == "compute" && desc.Path[1] == "layer" {
				return status.Errorf(codes.Unimplemented,
					"the tensor forward operation is not implemented by this worker")
			}
		}
	}
}

// GetSchema returns the Arrow schema for the requested generation stream.
func (s *InferenceFlightServer) GetSchema(ctx context.Context, desc *flight.FlightDescriptor) (*flight.SchemaResult, error) {
	meta := arrow.NewMetadata([]string{"QUARREL:device_id"}, []string{"0"})

	// It has to be the same schema the DoGet stream carries, because a
	// client that reads the schema before the data uses it to find the
	// columns.
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "token_id", Type: arrow.PrimitiveTypes.Int32},
		{Name: "logits", Type: arrow.FixedSizeListOf(10, arrow.PrimitiveTypes.Float32), Nullable: true},
		{Name: "text", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "id", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "vector", Type: arrow.FixedSizeListOf(1, arrow.PrimitiveTypes.Float32), Nullable: true},
	}, &meta)
	return &flight.SchemaResult{
		Schema: flight.SerializeSchema(schema, s.allocator),
	}, nil
}

// GetFlightInfo returns the available schema and endpoints for generation streams.
func (s *InferenceFlightServer) GetFlightInfo(ctx context.Context, desc *flight.FlightDescriptor) (*flight.FlightInfo, error) {
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "token_id", Type: arrow.PrimitiveTypes.Int32},
	}, nil)
	return &flight.FlightInfo{
		Schema:           flight.SerializeSchema(schema, s.allocator),
		FlightDescriptor: desc,
		Endpoint: []*flight.FlightEndpoint{
			{Ticket: &flight.Ticket{Ticket: []byte("generation_stream")}},
		},
	}, nil
}

// Serve starts the gRPC Flight Server on the configured address.
func (s *InferenceFlightServer) Serve() error {
	lis, err := net.Listen("tcp", s.addr)
	if err != nil {
		return fmt.Errorf("failed to listen: %v", err)
	}
	return s.ServeListener(lis)
}

// ServeListener starts the gRPC Flight Server using an existing listener.
func (s *InferenceFlightServer) ServeListener(lis net.Listener) error {
	grpcServer := grpc.NewServer()
	flight.RegisterFlightServiceServer(grpcServer, s)

	logger.Log.Info("Starting Arrow Flight Inference stream", "addr", lis.Addr().String())
	return grpcServer.Serve(lis)
}
