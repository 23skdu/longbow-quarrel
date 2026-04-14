package arrow_client

import (
	"context"
	"fmt"
	"net"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/flight"
	"google.golang.org/grpc"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

// MockServer implements a basic Flight server for integration testing
type MockServer struct {
	flight.BaseFlightServer
	ReceivedRecords []arrow.Record
	LastSchema      *arrow.Schema
}

func (s *MockServer) DoPut(stream flight.FlightService_DoPutServer) error {
	reader, err := flight.NewRecordReader(stream)
	if err != nil {
		return err
	}
	defer reader.Release()

	s.LastSchema = reader.Schema()

	for reader.Next() {
		record := reader.Record()
		record.Retain()
		s.ReceivedRecords = append(s.ReceivedRecords, record)
	}

	return stream.SendMsg(&flight.PutResult{})
}

func TestStreamEmbeddingsIntegration(t *testing.T) {
	// 1. Start a real in-process Flight server
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	addr := lis.Addr().String()

	grpcServer := grpc.NewServer()
	mockSrv := &MockServer{}
	flight.RegisterFlightServiceServer(grpcServer, mockSrv)

	go func() {
		_ = grpcServer.Serve(lis)
	}()
	defer grpcServer.Stop()

	// 2. Setup Client
	host, port, _ := net.SplitHostPort(addr)
	p := 0
	fmt.Sscanf(port, "%d", &p)
	client, err := NewFlightClient(host, p, host, p+1) // Meta port unused here
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	ctx := context.Background()
	if err := client.Connect(ctx); err != nil {
		t.Fatalf("failed to connect: %v", err)
	}
	defer client.Close()

	// 3. Prepare Test Data (Device Tensors)
	devCtx := device.NewContext()
	defer devCtx.Free()

	// Two tensors, different sizes, same type
	t1 := devCtx.NewTensorWithType(2, 4, device.DataTypeF32)
	t2 := devCtx.NewTensorWithType(3, 4, device.DataTypeF32)
	defer t1.Free()
	defer t2.Free()

	_ = t1.LoadFrom([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	_ = t2.LoadFrom([]float32{9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20})

	ids := []string{"id1", "id2", "id3", "id4", "id5"}

	// 4. Execute StreamEmbeddings
	err = client.StreamEmbeddings(ctx, []*device.Tensor{t1, t2}, ids, nil)
	if err != nil {
		t.Fatalf("StreamEmbeddings failed: %v", err)
	}

	// 5. Verify results on server
	if len(mockSrv.ReceivedRecords) != 2 {
		t.Errorf("expected 2 records, got %d", len(mockSrv.ReceivedRecords))
	}

	// Verify Schema
	if mockSrv.LastSchema == nil {
		t.Fatal("no schema received")
	}
	if mockSrv.LastSchema.Field(1).Type.ID() != arrow.FIXED_SIZE_LIST {
		t.Errorf("expected fixed size list, got %v", mockSrv.LastSchema.Field(1).Type)
	}

	// Verify Data
	totalRows := 0
	for _, r := range mockSrv.ReceivedRecords {
		totalRows += int(r.NumRows())
	}
	if totalRows != 5 {
		t.Errorf("expected 5 total rows, got %d", totalRows)
	}

	// Verify IDs from first record
	idsCol := mockSrv.ReceivedRecords[0].Column(0).(*array.String)
	if idsCol.Value(0) != "id1" || idsCol.Value(1) != "id2" {
		t.Errorf("incorrect IDs in first record: %s, %s", idsCol.Value(0), idsCol.Value(1))
	}
	// 6. Test F16 Streaming
	t.Run("F16Streaming", func(t *testing.T) {
		mockSrv.ReceivedRecords = nil // Reset
		
		t1_16 := devCtx.NewTensorWithType(1, 4, device.DataTypeF16)
		defer t1_16.Free()

		err = client.StreamEmbeddings(ctx, []*device.Tensor{t1_16}, []string{"f16_id"}, nil)
		if err != nil {
			t.Fatalf("F16 StreamEmbeddings failed: %v", err)
		}

		if len(mockSrv.ReceivedRecords) != 1 {
			t.Errorf("expected 1 record, got %d", len(mockSrv.ReceivedRecords))
		}
		if mockSrv.LastSchema.Field(1).Type.(*arrow.FixedSizeListType).Elem().ID() != arrow.FLOAT16 {
			t.Errorf("expected Float16 element, got %v", mockSrv.LastSchema.Field(1).Type)
		}
	})
}
