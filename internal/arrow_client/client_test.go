package arrow_client

import (
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
}

func TestDoPutReturnsErrorWhenNotConnected(t *testing.T) {
	client, _ := NewFlightClient("localhost", 3000, "localhost", 3001)

	vectors := [][]float32{{0.1, 0.2, 0.3}}

	err := client.DoPut(nil, vectors, nil, nil)

	if err == nil {
		t.Error("Expected error when client not connected")
	}
	if !strings.Contains(err.Error(), "not connected") {
		t.Errorf("Expected 'not connected' error, got: %v", err)
	}
}
