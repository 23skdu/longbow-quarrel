package arrow_client_test

import (
	"context"
	"fmt"
	"net"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/api"
	"github.com/23skdu/longbow-quarrel/internal/arrow_client"
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func TestArrowIntegration_Suite(t *testing.T) {
	// 1. Start InferenceFlightServer on random port
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	addr := lis.Addr().String()
	host, portStr, _ := net.SplitHostPort(addr)
	var port int
	fmt.Sscanf(portStr, "%d", &port)

	mockEngine, _ := engine.NewMockEngine("", config.Config{})
	mockTokenizer := &tokenizer.Tokenizer{
		Vocab:  map[string]int{" ": 0, "test": 1},
		Tokens: []string{" ", "test"},
	}
	server := api.NewInferenceFlightServer(addr, mockEngine, mockTokenizer)
	
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Run server in bg
	go func() {
		_ = server.ServeListener(lis)
	}()

	// Wait for port
	time.Sleep(500 * time.Millisecond)

	// 2. Connect client
	client, err := arrow_client.NewFlightClient(host, port, host, port)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	if err := client.Connect(ctx); err != nil {
		t.Fatalf("failed to connect: %v", err)
	}
	defer client.Close()

	// 3. Test DoGet
	t.Run("DoGet", func(t *testing.T) {
		batch, err := client.DoGet(ctx, []string{"test_seq"})
		if err != nil {
			t.Fatalf("DoGet failed: %v", err)
		}
		if len(batch.Vectors) > 100 { // Just some sanity check
			t.Errorf("unexpected batch size")
		}
	})

	// 4. Test GetSchema
	t.Run("GetSchema", func(t *testing.T) {
		schema, err := client.GetSchema(ctx)
		if err != nil {
			t.Fatalf("GetSchema failed: %v", err)
		}
		if schema == nil || len(schema.Fields()) == 0 {
			t.Errorf("invalid schema")
		}
	})

	// 5. Test GetFlightInfo
	t.Run("GetFlightInfo", func(t *testing.T) {
		info, err := client.GetFlightInfo(ctx, "embeddings")
		if err != nil {
			t.Fatalf("GetFlightInfo failed: %v", err)
		}
		if info == nil {
			t.Errorf("invalid flight info")
		}
	})

	// 6. Test DoPut
	t.Run("DoPut", func(t *testing.T) {
		vectors := [][]float32{{0.1, 0.2}}
		ids := []string{"id1"}
		meta := map[string]string{"key": "val"}
		if err := client.DoPut(ctx, vectors, ids, meta); err != nil {
			t.Fatalf("DoPut failed: %v", err)
		}
	})
}
