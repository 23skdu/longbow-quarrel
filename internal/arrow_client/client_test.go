package arrow_client

import (
	"context"
	"strings"
	"testing"
)

func TestNewFlightClient(t *testing.T) {
	client, err := NewFlightClient("localhost", 3000, "localhost", 3001)
	if err != nil {
		t.Fatalf("Failed to create FlightClient: %v", err)
	}

	if client == nil {
		t.Fatal("Expected non-nil client")
	}

	// Basic structural test
	if client.dataAddr == "" {
		t.Error("Expected dataAddr to be set")
	}
	if client.metaAddr == "" {
		t.Error("Expected metaAddr to be set")
	}
	if client.dataAddr != "localhost:3000" {
		t.Errorf("Expected dataAddr to be localhost:3000, got %s", client.dataAddr)
	}
	if client.metaAddr != "localhost:3001" {
		t.Errorf("Expected metaAddr to be localhost:3001, got %s", client.metaAddr)
	}
}

func TestNewFlightClientDefaults(t *testing.T) {
	client, err := NewFlightClient("127.0.0.1", 0, "127.0.0.1", 0)
	if err != nil {
		t.Fatalf("Failed to create FlightClient: %v", err)
	}

	if client.dataAddr != "127.0.0.1:3000" {
		t.Errorf("Expected default data port 3000, got %s", client.dataAddr)
	}
	if client.metaAddr != "127.0.0.1:3001" {
		t.Errorf("Expected default meta port 3001, got %s", client.metaAddr)
	}
}

func TestNewRecordBatch(t *testing.T) {
	vectors := [][]float32{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	}
	ids := []string{"doc1", "doc2"}

	batch := NewRecordBatch(vectors, ids)

	if batch == nil {
		t.Fatal("Expected non-nil batch")
	}
	if len(batch.Vectors) != 2 {
		t.Errorf("Expected 2 vectors, got %d", len(batch.Vectors))
	}
	if len(batch.Ids) != 2 {
		t.Errorf("Expected 2 ids, got %d", len(batch.Ids))
	}
	if batch.Metadata == nil {
		t.Error("Expected non-nil Metadata map")
	}
}

func TestDoPutReturnsErrorWhenNotConnected(t *testing.T) {
	client, _ := NewFlightClient("localhost", 3000, "localhost", 3001)

	vectors := [][]float32{{0.1, 0.2, 0.3}}

	ctx := context.Background()
	err := client.DoPut(ctx, vectors, nil, nil)

	if err == nil {
		t.Error("Expected error when client not connected")
	}
	if !strings.Contains(err.Error(), "not connected") {
		t.Errorf("Expected 'not connected' error, got: %v", err)
	}
}

func TestDoGetReturnsErrorWhenNotConnected(t *testing.T) {
	client, _ := NewFlightClient("localhost", 3000, "localhost", 3001)

	ctx := context.Background()
	_, err := client.DoGet(ctx, []string{"doc1"})

	if err == nil {
		t.Error("Expected error when client not connected")
	}
	if !strings.Contains(err.Error(), "not connected") {
		t.Errorf("Expected 'not connected' error, got: %v", err)
	}
}

func TestGetSchemaReturnsErrorWhenNotConnected(t *testing.T) {
	client, _ := NewFlightClient("localhost", 3000, "localhost", 3001)

	ctx := context.Background()
	_, err := client.GetSchema(ctx)

	if err == nil {
		t.Error("Expected error when client not connected")
	}
	if !strings.Contains(err.Error(), "not connected") {
		t.Errorf("Expected 'not connected' error, got: %v", err)
	}
}

// TestDoPutEmptyVectors checks that a message with no vectors is a
// control message rather than a malformed request.
//
// The remote worker uses it for the operations that carry no tensor, such
// as loading an adapter and synchronising the weights. The old check
// rejected them before they ever reached the worker, so those operations
// could never succeed.
func TestDoPutEmptyVectors(t *testing.T) {
	client, _ := NewFlightClient("localhost", 3000, "localhost", 3001)

	ctx := context.Background()
	err := client.DoPut(ctx, [][]float32{}, nil, nil)

	// The client is not connected, so the control message cannot be
	// sent, but it is not rejected for the wrong reason.
	if err == nil {
		t.Error("Expected an error for a disconnected client")
	}
	if strings.Contains(err.Error(), "no vectors") {
		t.Errorf("An empty vector list is a control message, not an error: %v", err)
	}
}

func TestDoPutZeroDimensionVectors(t *testing.T) {
	client, _ := NewFlightClient("localhost", 3000, "localhost", 3001)

	vectors := [][]float32{{}}
	ctx := context.Background()
	err := client.DoPut(ctx, vectors, nil, nil)

	if err == nil {
		t.Error("Expected error for zero-dimension vectors")
	}
	if !strings.Contains(err.Error(), "zero dimension") {
		t.Errorf("Expected 'zero dimension' error, got: %v", err)
	}
}

func TestDoPutMismatchedDimensions(t *testing.T) {
	client, _ := NewFlightClient("localhost", 3000, "localhost", 3001)

	vectors := [][]float32{
		{0.1, 0.2, 0.3},
		{0.4, 0.5}, // Different dimension
	}
	ctx := context.Background()
	err := client.DoPut(ctx, vectors, nil, nil)

	if err == nil {
		t.Error("Expected error for mismatched vector dimensions")
	}
}

