package api

import (
	"fmt"
	"net"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/flight"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"google.golang.org/grpc"
)

// InferenceFlightServer handles bidirectional Arrow streaming for the generation endpoints.
type InferenceFlightServer struct {
	flight.BaseFlightServer
	allocator memory.Allocator
	addr      string
}

// NewInferenceFlightServer initializes the Arrow Flight generation server.
func NewInferenceFlightServer(addr string) *InferenceFlightServer {
	return &InferenceFlightServer{
		allocator: memory.NewGoAllocator(),
		addr:      addr,
	}
}

// DoGet streams generated tokens directly using Arrow Array formatting.
func (s *InferenceFlightServer) DoGet(tckt *flight.Ticket, stream flight.FlightService_DoGetServer) error {
	// The ticket contains the sequence ID / context to attach to.
	// Schema: [token_id: int32, logits: list<float32>, metadata: string]
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "token_id", Type: arrow.PrimitiveTypes.Int32},
		{Name: "logits", Type: arrow.FixedSizeListOf(int32(10), arrow.PrimitiveTypes.Float32), Nullable: true},
		{Name: "metadata", Type: arrow.BinaryTypes.String, Nullable: true},
	}, nil)

	writer := flight.NewRecordWriter(stream, ipc.WithSchema(schema))
	defer writer.Close()

	builder := array.NewRecordBuilder(s.allocator, schema)
	defer builder.Release()

	// In real execution, this would be tied to a generation channel.
	// We'll prepare the builder and wait for token events.
	// tokenIDBuilder := builder.Field(0).(*array.Int32Builder)
	// logitsListBuilder := builder.Field(1).(*array.FixedSizeListBuilder)
	// metadataBuilder := builder.Field(2).(*array.StringBuilder)

	return nil
}


// Serve starts the gRPC Flight Server.
func (s *InferenceFlightServer) Serve() error {
	lis, err := net.Listen("tcp", s.addr)
	if err != nil {
		return fmt.Errorf("failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	flight.RegisterFlightServiceServer(grpcServer, s)

	fmt.Printf("Starting Arrow Flight Inference stream on %s\n", s.addr)
	return grpcServer.Serve(lis)
}
