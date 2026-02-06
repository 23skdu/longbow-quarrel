package arrow_client

import (
	"context"
	"fmt"
	"sync"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/flight"
)

// MockFlightClient is a mock implementation for testing
type MockFlightClient struct {
	mu        sync.RWMutex
	connected bool
	data      map[string]*RecordBatch
	schema    *arrow.Schema
}

// NewMockFlightClient creates a new mock client
func NewMockFlightClient() *MockFlightClient {
	return &MockFlightClient{
		data: make(map[string]*RecordBatch),
	}
}

// Connect simulates connection
func (m *MockFlightClient) Connect(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.connected = true
	return nil
}

// Close simulates disconnection
func (m *MockFlightClient) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.connected = false
	return nil
}

// DoPut stores vectors in memory
func (m *MockFlightClient) DoPut(ctx context.Context, vectors [][]float32, ids []string, metadata map[string]string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.connected {
		return fmt.Errorf("client not connected")
	}

	// Use first ID as key, or generate one
	key := "default"
	if len(ids) > 0 {
		key = ids[0]
	}

	m.data[key] = &RecordBatch{
		Vectors:  vectors,
		Ids:      ids,
		Metadata: metadata,
	}

	return nil
}

// DoGet retrieves vectors from memory
func (m *MockFlightClient) DoGet(ctx context.Context, ids []string) (*RecordBatch, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.connected {
		return nil, fmt.Errorf("client not connected")
	}

	if len(ids) == 0 {
		// Return all data
		var allVectors [][]float32
		var allIds []string
		for _, batch := range m.data {
			allVectors = append(allVectors, batch.Vectors...)
			allIds = append(allIds, batch.Ids...)
		}
		return &RecordBatch{
			Vectors:  allVectors,
			Ids:      allIds,
			Metadata: make(map[string]string),
		}, nil
	}

	// Return specific IDs
	key := ids[0]
	if batch, ok := m.data[key]; ok {
		return batch, nil
	}

	return nil, fmt.Errorf("not found")
}

// GetSchema returns mock schema
func (m *MockFlightClient) GetSchema(ctx context.Context) (*arrow.Schema, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.connected {
		return nil, fmt.Errorf("client not connected")
	}

	return m.schema, nil
}

// GetFlightInfo returns mock flight info
func (m *MockFlightClient) GetFlightInfo(ctx context.Context, path ...string) (*flight.FlightInfo, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.connected {
		return nil, fmt.Errorf("client not connected")
	}

	// Return empty flight info
	return &flight.FlightInfo{}, nil
}

// GetStoredData returns all stored data (for testing)
func (m *MockFlightClient) GetStoredData() map[string]*RecordBatch {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string]*RecordBatch)
	for k, v := range m.data {
		result[k] = v
	}
	return result
}

// Reset clears all stored data
func (m *MockFlightClient) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.data = make(map[string]*RecordBatch)
}
