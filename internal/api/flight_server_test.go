package api

import (
	"context"
	"net"
	"testing"
	"time"

	"github.com/apache/arrow-go/v18/arrow/flight"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func TestInferenceFlightServer_DoGet(t *testing.T) {
	// Find a free port
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}

	serverAddr := l.Addr().String()
	mockEngine := &engine.MockEngine{}
	mockTok := &tokenizer.Tokenizer{}
	server := NewInferenceFlightServer(serverAddr, mockEngine, mockTok)
	
	grpcServer := grpc.NewServer()
	flight.RegisterFlightServiceServer(grpcServer, server)

	// Start server in background
	go func() {
		_ = grpcServer.Serve(l)
	}()
	time.Sleep(100 * time.Millisecond) // wait for server to start

	defer grpcServer.Stop()

	// Client code to connect and pull tokens
	client, err := flight.NewFlightClient(serverAddr, nil, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("failed to create flight client: %v", err)
	}
	defer client.Close()

	ticket := &flight.Ticket{Ticket: []byte("test-sequence-id")}
	
	stream, err := client.DoGet(context.Background(), ticket)
	if err != nil {
		t.Fatalf("failed to execute DoGet: %v", err)
	}

	reader, err := flight.NewRecordReader(stream)
	if err != nil {
		t.Fatalf("failed to create generic record reader: %v", err)
	}
	defer reader.Release()

	// We expect the mock to close immediately right now because the loop is empty, 
	// but the fundamental gRPC connection and Arrow schema handshake succeed.
	if reader.Next() {
		rec := reader.Record()
		if rec.NumRows() == 0 {
			t.Errorf("expected > 0 rows")
		}
	}
}
