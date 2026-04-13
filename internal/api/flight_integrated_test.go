package api

import (
	"context"
	"net"
	"strconv"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/arrow_client"
)

func TestFlight_Integrated_Coverage(t *testing.T) {
	// 1. Start InferenceFlightServer on random port
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	addr := lis.Addr().String()
	host, portStr, err := net.SplitHostPort(addr)
	if err != nil {
		t.Fatalf("failed to split host port: %v", err)
	}
	port, _ := strconv.Atoi(portStr)

	server := NewInferenceFlightServer(addr)
	
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Run server in bg
	go func() {
		_ = server.ServeListener(lis)
	}()

	// Wait for port
	time.Sleep(200 * time.Millisecond)

	// 2. Connect client (from arrow_client package)
	client, err := arrow_client.NewFlightClient(host, port, host, port)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	if err := client.Connect(ctx); err != nil {
		t.Fatalf("failed to connect: %v", err)
	}
	defer client.Close()

	// 3. Exercise API methods
	t.Run("Discovery", func(t *testing.T) {
		// Hits GetSchema and GetFlightInfo in flight_server.go
		_, _ = client.GetSchema(ctx)
		_, _ = client.GetFlightInfo(ctx, "test")
	})

	t.Run("Data", func(t *testing.T) {
		// Hits DoPut and DoGet in flight_server.go
		_, _ = client.DoGet(ctx, []string{"seq"})
		vectors := [][]float32{{0.1, 0.2}}
		ids := []string{"id"}
		meta := map[string]string{"k": "v"}
		_ = client.DoPut(ctx, vectors, ids, meta)
	})
}
