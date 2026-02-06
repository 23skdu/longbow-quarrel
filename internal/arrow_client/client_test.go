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

func TestDoPutEmptyVectors(t *testing.T) {
	client, _ := NewFlightClient("localhost", 3000, "localhost", 3001)

	ctx := context.Background()
	err := client.DoPut(ctx, [][]float32{}, nil, nil)

	if err == nil {
		t.Error("Expected error for empty vectors")
	}
	if !strings.Contains(err.Error(), "no vectors") {
		t.Errorf("Expected 'no vectors' error, got: %v", err)
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

// Mock Client Tests

func TestMockFlightClient(t *testing.T) {
	mock := NewMockFlightClient()
	ctx := context.Background()

	// Test not connected
	err := mock.DoPut(ctx, [][]float32{{0.1, 0.2}}, []string{"doc1"}, nil)
	if err == nil || !strings.Contains(err.Error(), "not connected") {
		t.Errorf("Expected 'not connected' error, got: %v", err)
	}

	// Connect
	err = mock.Connect(ctx)
	if err != nil {
		t.Fatalf("Failed to connect mock: %v", err)
	}

	// Test DoPut
	vectors := [][]float32{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	}
	ids := []string{"doc1", "doc2"}
	metadata := map[string]string{"model": "test"}

	err = mock.DoPut(ctx, vectors, ids, metadata)
	if err != nil {
		t.Errorf("DoPut failed: %v", err)
	}

	// Test DoGet
	batch, err := mock.DoGet(ctx, []string{"doc1"})
	if err != nil {
		t.Errorf("DoGet failed: %v", err)
	}
	if len(batch.Vectors) != 2 {
		t.Errorf("Expected 2 vectors, got %d", len(batch.Vectors))
	}

	// Test GetAll
	batch, err = mock.DoGet(ctx, nil)
	if err != nil {
		t.Errorf("DoGet all failed: %v", err)
	}
	if len(batch.Vectors) != 2 {
		t.Errorf("Expected 2 vectors in all batch, got %d", len(batch.Vectors))
	}

	// Test Close
	err = mock.Close()
	if err != nil {
		t.Errorf("Close failed: %v", err)
	}

	// Test GetSchema when connected
	mock.Connect(ctx)
	schema, err := mock.GetSchema(ctx)
	if err != nil {
		t.Errorf("GetSchema failed: %v", err)
	}
	if schema != nil {
		t.Error("Expected nil schema from mock")
	}
}

func TestMockFlightClientReset(t *testing.T) {
	mock := NewMockFlightClient()
	ctx := context.Background()
	mock.Connect(ctx)

	// Add data
	mock.DoPut(ctx, [][]float32{{0.1, 0.2}}, []string{"doc1"}, nil)

	// Verify data exists
	data := mock.GetStoredData()
	if len(data) != 1 {
		t.Errorf("Expected 1 entry, got %d", len(data))
	}

	// Reset
	mock.Reset()

	// Verify data cleared
	data = mock.GetStoredData()
	if len(data) != 0 {
		t.Errorf("Expected 0 entries after reset, got %d", len(data))
	}
}
