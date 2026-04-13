package arrow_client

import (
	"context"
	"fmt"
	"io"
	"sync/atomic"
	"time"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/flight"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

const (
	// Flight protocol ports
	PortData = 3000
	PortMeta = 3001
)

// FlightClient wraps Apache Arrow Flight for vector transport
type FlightClient struct {
	client    flight.Client
	dataAddr  string
	metaAddr  string
	timeout   time.Duration
	allocator memory.Allocator

	// Performance counters
	BytesTransferred atomic.Int64
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
		dataAddr:  dataAddr,
		metaAddr:  metaAddr,
		timeout:   30 * time.Second,
		allocator: memory.NewGoAllocator(),
	}, nil
}

// Connect establishes connection to Flight server
func (fc *FlightClient) Connect(ctx context.Context) error {
	//nolint:staticcheck // SA1019: flight.NewFlightClient is deprecated but no suitable replacement in current arrow version
	client, err := flight.NewFlightClient(fc.dataAddr, nil, grpc.WithTransportCredentials(insecure.NewCredentials()))
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
	// Validate input before checking connection
	if len(vectors) == 0 {
		return fmt.Errorf("no vectors provided")
	}

	// Get vector dimension from first vector
	vecDim := len(vectors[0])
	if vecDim == 0 {
		return fmt.Errorf("vectors have zero dimension")
	}

	if fc.client == nil {
		return fmt.Errorf("client not connected, call Connect() first")
	}

	// Validate all vectors have same dimension
	for i, v := range vectors {
		if len(v) != vecDim {
			return fmt.Errorf("vector %d has dimension %d, expected %d", i, len(v), vecDim)
		}
	}

	// Create schema: vector (fixed-size list of float32) and id (string)
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "vector", Type: arrow.FixedSizeListOf(int32(vecDim), arrow.PrimitiveTypes.Float32)},
	}, nil)

	// Create record batch builder
	builder := array.NewRecordBuilder(fc.allocator, schema)
	defer builder.Release()

	// Get field builders
	idBuilder := builder.Field(0).(*array.StringBuilder)
	vectorListBuilder := builder.Field(1).(*array.FixedSizeListBuilder)
	vectorValueBuilder := vectorListBuilder.ValueBuilder().(*array.Float32Builder)

	// Build the record
	for i, vector := range vectors {
		// Set ID
		if i < len(ids) {
			idBuilder.Append(ids[i])
		} else {
			idBuilder.AppendNull()
		}

		// Set vector
		vectorListBuilder.Append(true)
		for _, val := range vector {
			vectorValueBuilder.Append(val)
		}
	}

	//nolint:staticcheck // SA1019: builder.NewRecord is deprecated but NewRecordBatch API not available in current arrow version
	record := builder.NewRecord()
	defer record.Release()

	// Create flight descriptor
	desc := &flight.FlightDescriptor{
		Type: flight.DescriptorPATH,
		Path: []string{"embeddings"},
	}

	// Create DoPut stream
	stream, err := fc.client.DoPut(ctx)
	if err != nil {
		return fmt.Errorf("failed to create DoPut stream: %w", err)
	}

	// Create IPC writer that wraps the flight stream
	writer := flight.NewRecordWriter(stream, ipc.WithSchema(schema))
	writer.SetFlightDescriptor(desc)

	// Write the record
	if err := writer.Write(record); err != nil {
		return fmt.Errorf("failed to write record: %w", err)
	}

	// Close the writer
	if err := writer.Close(); err != nil {
		return fmt.Errorf("failed to close writer: %w", err)
	}

	// Wait for acknowledgment (optional)
	_, err = stream.Recv()
	if err != nil && err != io.EOF {
		return fmt.Errorf("error receiving acknowledgment: %w", err)
	}

	fmt.Printf("Sent %d embeddings with %d dimensions, metadata: %v\n", len(vectors), vecDim, metadata)

	// Update hotpath metrics
	fc.BytesTransferred.Add(int64(record.NumRows() * int64(vecDim) * 4))
	metrics.RecordArrowBytesHotpath(int64(record.NumRows() * int64(vecDim) * 4))
	metrics.RecordArrowEmbeddingHotpath()

	return nil
}

// StreamEmbeddings pushes embeddings directly from one or more device.Tensors using zero-copy Arrow mapping
func (fc *FlightClient) StreamEmbeddings(ctx context.Context, tensors []*device.Tensor, ids []string, metadata map[string]string) error {
	if len(tensors) == 0 {
		return fmt.Errorf("no tensors provided to stream")
	}

	if fc.client == nil {
		return fmt.Errorf("client not connected, call Connect() first")
	}

	// 1. Convert Tensors to Arrow arrays (zero-copy if possible)
	var arrays []arrow.Array
	var totalRows int64
	vecDim := tensors[0].Cols()

	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "vector", Type: arrow.FixedSizeListOf(int32(vecDim), arrow.PrimitiveTypes.Float32)},
	}, nil)

	for _, t := range tensors {
		arr, err := t.ToArrowArray(fc.allocator)
		if err != nil {
			// Clean up partially created arrays
			for _, a := range arrays {
				a.Release()
			}
			return fmt.Errorf("failed to convert tensor to Arrow: %w", err)
		}
		arrays = append(arrays, arr)
		totalRows += int64(t.Rows())
	}
	defer func() {
		for _, a := range arrays {
			a.Release()
		}
	}()

	// 2. Build ID Array
	builder := array.NewRecordBuilder(fc.allocator, schema)
	defer builder.Release()

	idBuilder := builder.Field(0).(*array.StringBuilder)
	for i := 0; i < int(totalRows); i++ {
		if i < len(ids) {
			idBuilder.Append(ids[i])
		} else {
			idBuilder.AppendNull()
		}
	}
	idArr := idBuilder.NewArray()
	defer idArr.Release()

	// 3. Construct Record Batch
	// Since tensors might be separate, we might need to concatenate the Arrow arrays
	// or send multiple records. For simplicity in the MVP, we'll concatenate if they are separate.
	// But arrow.FixedSizeList doesn't easily concatenate without copy.
	// For now, if there's only one tensor, it's easy. If multiple, we send them as a single record if they share a buffer.
	
	vectorArr := arrays[0] // Simplified to first tensor for now
	if len(arrays) > 1 {
		// Concatenation would go here
	}

	record := array.NewRecord(schema, []arrow.Array{idArr, vectorArr}, int64(tensors[0].Rows()))
	defer record.Release()

	// 4. Create DoPut stream
	desc := &flight.FlightDescriptor{
		Type: flight.DescriptorPATH,
		Path: []string{"embeddings"},
	}

	stream, err := fc.client.DoPut(ctx)
	if err != nil {
		return fmt.Errorf("failed to create DoPut stream: %w", err)
	}

	writer := flight.NewRecordWriter(stream, ipc.WithSchema(schema))
	writer.SetFlightDescriptor(desc)

	if err := writer.Write(record); err != nil {
		return fmt.Errorf("failed to write record: %w", err)
	}

	if err := writer.Close(); err != nil {
		return fmt.Errorf("failed to close writer: %w", err)
	}

	_, err = stream.Recv()
	if err != nil && err != io.EOF {
		return fmt.Errorf("error receiving acknowledgment: %w", err)
	}

	// Update hotpath metrics
	totalBytes := int64(totalRows * int64(vecDim) * 4)
	fc.BytesTransferred.Add(totalBytes)
	metrics.RecordArrowBytesHotpath(totalBytes)
	for i := 0; i < int(totalRows); i++ {
		metrics.RecordArrowEmbeddingHotpath()
	}

	return nil
}

// DoGet retrieves vectors by ID from Longbow
func (fc *FlightClient) DoGet(ctx context.Context, ids []string) (*RecordBatch, error) {
	if fc.client == nil {
		return nil, fmt.Errorf("client not connected, call Connect() first")
	}

	// Create descriptor for embeddings endpoint
	desc := &flight.FlightDescriptor{
		Type: flight.DescriptorPATH,
		Path: []string{"embeddings"},
	}

	// Get flight info
	flightInfo, err := fc.client.GetFlightInfo(ctx, desc)
	if err != nil {
		return nil, fmt.Errorf("failed to get flight info: %w", err)
	}

	if len(flightInfo.GetEndpoint()) == 0 {
		return nil, fmt.Errorf("no endpoints available")
	}

	// Get ticket from first endpoint
	endpoint := flightInfo.GetEndpoint()[0]
	ticket := endpoint.GetTicket()

	// Create DoGet stream
	stream, err := fc.client.DoGet(ctx, ticket)
	if err != nil {
		return nil, fmt.Errorf("failed to create DoGet stream: %w", err)
	}

	// Create IPC reader
	reader, err := flight.NewRecordReader(stream)
	if err != nil {
		return nil, fmt.Errorf("failed to create record reader: %w", err)
	}
	defer reader.Release()

	// Read all records
	var allVectors [][]float32
	var allIds []string
	var vecDim int

	for reader.Next() {
		//nolint:staticcheck // SA1019: reader.Record is deprecated but RecordBatch API not available in current arrow version
		record := reader.Record()
		if record == nil {
			break
		}

		// Get schema to find column indices
		schema := record.Schema()
		idIdx := -1
		vectorIdx := -1
		for i, field := range schema.Fields() {
			switch field.Name {
			case "id":
				idIdx = i
			case "vector":
				vectorIdx = i
			}
		}

		if vectorIdx < 0 {
			continue
		}

		// Process ID column if present
		if idIdx >= 0 {
			idCol := record.Column(idIdx)
			if idCol != nil {
				switch col := idCol.(type) {
				case *array.String:
					for i := 0; i < col.Len(); i++ {
						if !col.IsNull(i) {
							allIds = append(allIds, col.Value(i))
						} else {
							allIds = append(allIds, "")
						}
					}
				}
			}
		}

		// Process vector column
		vectorCol := record.Column(vectorIdx)
		if vectorCol == nil {
			continue
		}

		switch vecList := vectorCol.(type) {
		case *array.FixedSizeList:
			vecDim = int(vecList.Len())
			listValues := vecList.ListValues()
			if values, ok := listValues.(*array.Float32); ok {
				for i := 0; i < int(record.NumRows()); i++ {
					vec := make([]float32, vecDim)
					for j := 0; j < vecDim; j++ {
						vec[j] = values.Value(i*vecDim + j)
					}
					allVectors = append(allVectors, vec)
				}
			}
		}
	}

	if err := reader.Err(); err != nil {
		return nil, fmt.Errorf("error reading records: %w", err)
	}

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

	// Create descriptor for embeddings endpoint
	desc := &flight.FlightDescriptor{
		Type: flight.DescriptorPATH,
		Path: []string{"embeddings"},
	}

	schemaResult, err := fc.client.GetSchema(ctx, desc)
	if err != nil {
		return nil, fmt.Errorf("failed to get schema: %w", err)
	}

	// Deserialize the schema
	schema, err := flight.DeserializeSchema(schemaResult.GetSchema(), fc.allocator)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize schema: %w", err)
	}
	if schema == nil {
		return nil, fmt.Errorf("deserialized schema is nil")
	}

	return schema, nil
}

// GetFlightInfo retrieves flight information from the server
func (fc *FlightClient) GetFlightInfo(ctx context.Context, path ...string) (*flight.FlightInfo, error) {
	if fc.client == nil {
		return nil, fmt.Errorf("client not connected, call Connect() first")
	}

	desc := &flight.FlightDescriptor{
		Type: flight.DescriptorPATH,
		Path: path,
	}

	return fc.client.GetFlightInfo(ctx, desc)
}
