package api

import (
	"context"
	"fmt"
	"net"

	"github.com/apache/arrow-go/v18/arrow"
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
	// For MVP, we return a mock stream of generated tensors.
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "token_id", Type: arrow.PrimitiveTypes.Int32},
		{Name: "logits", Type: arrow.FixedSizeListOf(int32(10), arrow.PrimitiveTypes.Float32)},
	}, nil)

	writer := flight.NewRecordWriter(stream, ipc.WithSchema(schema))
	defer writer.Close()

	// In real execution, this loop bridges Engine tokens to the RPC writer natively.
	// We mocked generating 5 tokens here as proof-of-concept.
	for i := 0; i < 5; i++ {
		// Mock token generation logic hook
		// token, tensor := engine.GenerateStep()
	}

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
