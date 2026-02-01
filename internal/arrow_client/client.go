package arrow_client

import (
	"context"
	"fmt"
	"time"

	"github.com/apache/arrow/go/v16/arrow"
	"github.com/apache/arrow/go/v16/arrow/array"
	"github.com/apache/arrow/go/v16/arrow/flight"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	// Flight protocol ports
	PortData = 3000
	PortMeta = 3001
)

// FlightClient wraps Apache Arrow Flight for vector transport
type FlightClient struct {
	client   *flight.Client
	dataAddr string
	metaAddr string
	timeout  time.Duration
}

// NewFlightClient creates a new Flight client connection
func NewFlightClient(dataHost string, dataPort int, metaHost string, metaPort int) (*FlightClient, error) {
	if dataPort <= 0 {
		dataPort = PortData
	}
	if metaPort <= 0 {
		metaPort = PortMeta
	}

	dataAddr := fmt.Sprintf("%s:%d", dataHost, dataPort)
	metaAddr := fmt.Sprintf("%s:%d", metaHost, metaPort)

	return &FlightClient{
		dataAddr: dataAddr,
		metaAddr: metaAddr,
		timeout:  30 * time.Second,
	}, nil
}

// Connect establishes connection to Flight server
func (fc *FlightClient) Connect(ctx context.Context) error {
	client, err := flight.NewClient(ctx, "grpc://"+fc.dataAddr, insecure.NewCredentials(), grpc.WithBlock(false))
	if err != nil {
		return fmt.Errorf("failed to create Flight client: %w", err)
	}
	fc.client = client
	return nil
}

// Close disconnects from Flight server
func (fc *FlightClient) Close() error {
	if fc.client != nil {
		return fc.client.Close()
	}
	return nil
}

// RecordBatch represents a batch of vectors with metadata
type RecordBatch struct {
	Vectors  [][]float32       // Embedding vectors
	Ids      []string          // Optional: document or item IDs
	Metadata map[string]string // Additional metadata
}

// NewRecordBatch creates a new record batch from raw vectors
func NewRecordBatch(vectors [][]float32, ids []string) *RecordBatch {
	return &RecordBatch{
		Vectors:  vectors,
		Ids:      ids,
		Metadata: make(map[string]string),
	}
}

// DoPut sends embeddings to Longbow data port
func (fc *FlightClient) DoPut(ctx context.Context, vectors [][]float32, ids []string, metadata map[string]string) error {
	if fc.client == nil {
		return fmt.Errorf("client not connected, call Connect() first")
	}

	// Determine dimensions
	if len(vectors) == 0 {
		return fmt.Errorf("no vectors provided")
	}

	// Create schema
	schema := arrow.NewSchema([]arrow.Field{
		arrow.Field{Name: "vector", Type: arrow.FixedSizeListOf(arrow.Float32, int32(len(vectors[0])))},
		arrow.Field{Name: "id", Type: arrow.Binary},
	})

	// Create record with all vectors
	record := arrow.NewRecord(schema, int64(len(vectors)), nil)
	defer record.Release()

	// Convert vectors to Arrow array
	vectorBuilder := array.NewFloat64Builder(int32(len(vectors[0])))
	vectorBuilder.Reserve(int64(len(vectors)) * int64(len(vectors[0])))

	for _, vec := range vectors {
		vectorBuilder.AppendValues(vec)
	}

	// Create array
	vectorsArray := vectorBuilder.NewArray()

	// Set record columns
	record.SetColumn(0, vectorsArray)
	if ids != nil && len(ids) > 0 {
		// For now, skip ID column implementation
	}

	defer vectorsArray.Release()

	// Create descriptor
	desc := flight.NewDescriptor(
		"embeddings",
		[]string{"/data"},
	)
	if err != nil {
		return fmt.Errorf("failed to create descriptor: %w", err)
	}
	defer desc.Release()

	// Get flight info
	info, err := fc.client.GetFlightInfo(ctx, desc)
	if err != nil {
		return fmt.Errorf("failed to get flight info: %w", err)
	}
	defer info.Release()

	// Create writer
	writer, err := fc.client.DoPut(ctx, info)
	if err != nil {
		return fmt.Errorf("failed to create DoPut writer: %w", err)
	}
	defer writer.Release()

	// Write record
	if err := writer.WriteRecord(ctx, record); err != nil {
		return fmt.Errorf("failed to write record: %w", err)
	}

	// Close writer
	if err := writer.Close(); err != nil {
		return fmt.Errorf("failed to close writer: %w", err)
	}

	// Log result
	fmt.Printf("Sent %d embeddings with metadata: %v\n", len(vectors), metadata)

	return nil
}

// DoGet retrieves vectors by ID from Longbow
func (fc *FlightClient) DoGet(ctx context.Context, ids []string) (*RecordBatch, error) {
	if fc.client == nil {
		return nil, fmt.Errorf("client not connected, call Connect() first")
	}

	// Create descriptor for embeddings endpoint
	desc := flight.NewDescriptor(
		"embeddings",
		[]string{"/data"},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create descriptor: %w", err)
	}
	defer desc.Release()

	// Get flight info
	info, err := fc.client.GetFlightInfo(ctx, desc)
	if err != nil {
		return nil, fmt.Errorf("failed to get flight info: %w", err)
	}
	defer info.Release()

	// Create reader
	reader, err := fc.client.DoGet(ctx, info)
	if err != nil {
		return nil, fmt.Errorf("failed to create DoGet reader: %w", err)
	}
	defer reader.Release()

	// Read stream
	var allVectors [][]float32
	allIds := make([]string, 0)
	recordReader, err := reader.GetStream(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get stream: %w", err)
	}

	for recordReader.Next() {
		record := recordReader.Record()
		if record == nil {
			break
		}

		// Get vectors column (column 0)
		vectorsCol := record.Column(0)
		if vectorsCol == nil {
			continue
		}

		vectorsColRet := vectorsCol.(*array.Float64)
		vectorsColRet.Retain()

		// Convert to [][]float32
		embeddingDim := vectorsColRet.Len()
		vectorLen := int(vectorsColRet.Len())

		for i := 0; i < vectorLen; i += embeddingDim {
			row := make([]float32, embeddingDim)
			for j := 0; j < embeddingDim; j++ {
				row[j] = vectorsColRet.Value(i + j)
			}
			allVectors = append(allVectors, row)
		}
	}

	vectorsColRet.Release()

	recordReader.Release()

	return &RecordBatch{
		Vectors:  allVectors,
		Ids:      allIds,
		Metadata: make(map[string]string),
	}, nil
}

// GetSchema retrieves the schema from Longbow
func (fc *FlightClient) GetSchema(ctx context.Context) (*arrow.Schema, error) {
	if fc.client == nil {
		return nil, fmt.Errorf("client not connected, call Connect() first")
	}

	// Get flight info for embeddings endpoint
	desc := flight.NewDescriptor(
		"embeddings",
		[]string{"/data"},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create descriptor: %w", err)
	}
	defer desc.Release()

	info, err := fc.client.GetFlightInfo(ctx, desc)
	if err != nil {
		return nil, fmt.Errorf("failed to get flight info: %w", err)
	}
	defer info.Release()

	return info.Schema(), nil
}
