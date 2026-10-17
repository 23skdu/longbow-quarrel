package arrow_client

import (
	"context"
	"errors"
	"io"
	"net"
	"strings"
	"sync"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/flight"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

type testFlightServer struct {
	flight.BaseFlightServer
	mu            sync.Mutex
	records       []arrow.Record
	schema        *arrow.Schema
	hasEndpoint   bool
	sendPutRes    bool
	doGetRecords  []arrow.Record
	doPutErr      error
	doGetErr      error
	flightInfoErr       error
	schemaErr           error
	doGetErrAfterRecord error
}

func (s *testFlightServer) DoPut(stream flight.FlightService_DoPutServer) error {
	if s.doPutErr != nil {
		return s.doPutErr
	}
	reader, err := flight.NewRecordReader(stream)
	if err != nil {
		return err
	}
	defer reader.Release()
	for reader.Next() {
		rec := reader.Record()
		if rec != nil {
			rec.Retain()
			s.mu.Lock()
			s.records = append(s.records, rec)
			s.mu.Unlock()
		}
	}
	if err := reader.Err(); err != nil {
		return err
	}
	if s.sendPutRes {
		return stream.Send(&flight.PutResult{})
	}
	return nil
}

func (s *testFlightServer) DoGet(ticket *flight.Ticket, stream flight.FlightService_DoGetServer) error {
	if s.doGetErr != nil {
		return s.doGetErr
	}
	s.mu.Lock()
	records := make([]arrow.Record, len(s.doGetRecords))
	copy(records, s.doGetRecords)
	s.mu.Unlock()

	if len(records) > 0 {
		writer := flight.NewRecordWriter(stream, ipc.WithSchema(records[0].Schema()))
		for _, rec := range records {
			if err := writer.Write(rec); err != nil {
				_ = writer.Close()
				return err
			}
		}
		if err := writer.Close(); err != nil {
			return err
		}
	}
	if s.doGetErrAfterRecord != nil {
		return s.doGetErrAfterRecord
	}
	return nil
}

func (s *testFlightServer) GetFlightInfo(ctx context.Context, desc *flight.FlightDescriptor) (*flight.FlightInfo, error) {
	if s.flightInfoErr != nil {
		return nil, s.flightInfoErr
	}
	if !s.hasEndpoint {
		return &flight.FlightInfo{Endpoint: []*flight.FlightEndpoint{}}, nil
	}
	return &flight.FlightInfo{
		Endpoint: []*flight.FlightEndpoint{
			{Ticket: &flight.Ticket{Ticket: []byte("test-ticket")}},
		},
	}, nil
}

func (s *testFlightServer) GetSchema(ctx context.Context, desc *flight.FlightDescriptor) (*flight.SchemaResult, error) {
	if s.schemaErr != nil {
		return nil, s.schemaErr
	}
	if s.schema == nil {
		return &flight.SchemaResult{Schema: []byte("invalid-schema")}, nil
	}
	return &flight.SchemaResult{
		Schema: flight.SerializeSchema(s.schema, memory.DefaultAllocator),
	}, nil
}

func startTestFlightServer(t *testing.T, srv *testFlightServer) (*FlightClient, func()) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	flight.RegisterFlightServiceServer(s, srv)
	go func() {
		_ = s.Serve(lis)
	}()

	addr := lis.Addr().(*net.TCPAddr)
	client, err := NewFlightClient(addr.IP.String(), addr.Port, addr.IP.String(), addr.Port)
	if err != nil {
		s.Stop()
		_ = lis.Close()
		t.Fatalf("failed to create client: %v", err)
	}
	if err := client.Connect(context.Background()); err != nil {
		s.Stop()
		_ = lis.Close()
		t.Fatalf("failed to connect client: %v", err)
	}

	cleanup := func() {
		_ = client.Close()
		s.GracefulStop()
		_ = lis.Close()
		srv.mu.Lock()
		for _, r := range srv.records {
			r.Release()
		}
		srv.records = nil
		for _, r := range srv.doGetRecords {
			r.Release()
		}
		srv.doGetRecords = nil
		srv.mu.Unlock()
	}
	return client, cleanup
}

func TestMockFlightClient_AdditionalCoverage(t *testing.T) {
	ctx := context.Background()
	mock := NewMockFlightClient()

	// 1. DoGet when not connected
	if _, err := mock.DoGet(ctx, []string{"id1"}); err == nil || !strings.Contains(err.Error(), "not connected") {
		t.Errorf("expected not connected error from DoGet, got: %v", err)
	}

	// 2. Connect, then DoGet with non-existent id
	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("connect failed: %v", err)
	}
	if _, err := mock.DoGet(ctx, []string{"nonexistent"}); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("expected not found error from DoGet, got: %v", err)
	}

	// 3. Disconnect, then GetSchema
	if err := mock.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}
	if _, err := mock.GetSchema(ctx); err == nil || !strings.Contains(err.Error(), "not connected") {
		t.Errorf("expected not connected error from GetSchema, got: %v", err)
	}

	// 4. GetFlightInfo when not connected
	if _, err := mock.GetFlightInfo(ctx, "path1"); err == nil || !strings.Contains(err.Error(), "not connected") {
		t.Errorf("expected not connected error from GetFlightInfo, got: %v", err)
	}

	// 5. Connect, then GetFlightInfo
	if err := mock.Connect(ctx); err != nil {
		t.Fatalf("connect failed: %v", err)
	}
	info, err := mock.GetFlightInfo(ctx, "path1")
	if err != nil {
		t.Errorf("GetFlightInfo failed: %v", err)
	}
	if info == nil {
		t.Error("expected non-nil flight info")
	}
}

func TestFlightClient_Close_WhenNil(t *testing.T) {
	fc, err := NewFlightClient("127.0.0.1", 3000, "127.0.0.1", 3001)
	if err != nil {
		t.Fatalf("NewFlightClient failed: %v", err)
	}
	if err := fc.Close(); err != nil {
		t.Errorf("expected nil error on Close() when client is nil, got: %v", err)
	}
}

func TestFlightClient_Connect_Error(t *testing.T) {
	orig := flightNewClient
	defer func() { flightNewClient = orig }()

	flightNewClient = func(addr string, auth flight.ClientAuthHandler, opts ...grpc.DialOption) (flight.Client, error) {
		return nil, errors.New("mock dial failure")
	}

	fc, err := NewFlightClient("127.0.0.1", 3000, "127.0.0.1", 3001)
	if err != nil {
		t.Fatalf("NewFlightClient failed: %v", err)
	}
	err = fc.Connect(context.Background())
	if err == nil || !strings.Contains(err.Error(), "failed to create Flight client") {
		t.Errorf("expected failed to create Flight client error, got: %v", err)
	}
}

func TestFlightClient_DoPut_ServerInteractions(t *testing.T) {
	srv := &testFlightServer{}
	client, cleanup := startTestFlightServer(t, srv)
	defer cleanup()

	ctx := context.Background()

	// 1. Vector dimension mismatch when connected
	err := client.DoPut(ctx, [][]float32{{1.0, 2.0}, {3.0}}, []string{"id1", "id2"}, nil)
	if err == nil || !strings.Contains(err.Error(), "has dimension 1, expected 2") {
		t.Errorf("expected dimension mismatch error, got: %v", err)
	}

	// 2. Normal DoPut with fewer IDs than vectors (triggers AppendNull on idBuilder)
	vectors := [][]float32{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	}
	ids := []string{"id1"} // len(ids) < len(vectors)
	if err := client.DoPut(ctx, vectors, ids, map[string]string{"foo": "bar"}); err != nil {
		t.Fatalf("DoPut failed: %v", err)
	}

	if client.BytesTransferred.Load() == 0 {
		t.Error("expected BytesTransferred to be updated")
	}
}

func TestFlightClient_DoPutTensor_ServerInteractions(t *testing.T) {
	srv := &testFlightServer{}
	client, cleanup := startTestFlightServer(t, srv)
	defer cleanup()

	ctx := context.Background()

	// 1. len(data) == 0
	if _, err := client.DoPutTensor(ctx, nil, []int32{1}, nil); err == nil || !strings.Contains(err.Error(), "no data provided") {
		t.Errorf("expected no data provided error, got: %v", err)
	}

	// 2. Client not connected
	unconnClient, _ := NewFlightClient("127.0.0.1", 3000, "127.0.0.1", 3001)
	if _, err := unconnClient.DoPutTensor(ctx, []float32{1.0}, []int32{1}, nil); err == nil || !strings.Contains(err.Error(), "client not connected") {
		t.Errorf("expected client not connected error, got: %v", err)
	}

	// 3. Invalid tensor dimensions (len(data) < rowsCount so colsCount == 0)
	if _, err := client.DoPutTensor(ctx, []float32{1.0}, []int32{1, 1}, nil); err == nil || !strings.Contains(err.Error(), "invalid tensor dimensions") {
		t.Errorf("expected invalid tensor dimensions error, got: %v", err)
	}

	// 4. Default rowsCount = 1 when len(rows) == 0, with padding
	// data len=3, rows=[] -> rowsCount=1, colsCount=3
	res, err := client.DoPutTensor(ctx, []float32{1.0, 2.0, 3.0}, nil, nil)
	if err != nil {
		t.Fatalf("DoPutTensor failed with empty rows: %v", err)
	}
	if len(res) != 0 {
		t.Errorf("expected empty result, got %v", res)
	}

	// 5. Padding: rows=[2], data=[1, 2, 3] -> rowsCount=1, colsCount=3. rowLen=2, so offset=2, pads 1 zero.
	res, err = client.DoPutTensor(ctx, []float32{1.0, 2.0, 3.0}, []int32{2}, nil)
	if err != nil {
		t.Fatalf("DoPutTensor failed with padded row: %v", err)
	}

	// 6. With server sending PutResult response
	srv.mu.Lock()
	srv.sendPutRes = true
	srv.mu.Unlock()
	res, err = client.DoPutTensor(ctx, []float32{1.0, 2.0}, []int32{2}, nil)
	if err != nil {
		t.Fatalf("DoPutTensor failed with response: %v", err)
	}
}

func TestFlightClient_StreamEmbeddings_ServerInteractions(t *testing.T) {
	srv := &testFlightServer{}
	client, cleanup := startTestFlightServer(t, srv)
	defer cleanup()

	ctx := context.Background()

	// 1. len(tensors) == 0
	if err := client.StreamEmbeddings(ctx, nil, nil, nil); err == nil || !strings.Contains(err.Error(), "no tensors provided") {
		t.Errorf("expected no tensors provided error, got: %v", err)
	}

	// 2. Client not connected
	unconnClient, _ := NewFlightClient("127.0.0.1", 3000, "127.0.0.1", 3001)
	devCtx := device.NewContext()
	t1 := devCtx.NewTensorWithType(1, 2, device.DataTypeF32)
	if err := unconnClient.StreamEmbeddings(ctx, []*device.Tensor{t1}, nil, nil); err == nil || !strings.Contains(err.Error(), "client not connected") {
		t.Errorf("expected client not connected error, got: %v", err)
	}

	// 3. Unsupported tensor data type
	unsuppTensor := devCtx.NewTensorWithType(1, 4, device.DataTypeINT8)
	if err := client.StreamEmbeddings(ctx, []*device.Tensor{unsuppTensor}, nil, nil); err == nil || !strings.Contains(err.Error(), "unsupported tensor data type") {
		t.Errorf("expected unsupported tensor data type error, got: %v", err)
	}

	// 4. Tensor column mismatch
	tensorA := devCtx.NewTensorWithType(1, 4, device.DataTypeF32)
	tensorB := devCtx.NewTensorWithType(1, 2, device.DataTypeF32)
	if err := client.StreamEmbeddings(ctx, []*device.Tensor{tensorA, tensorB}, nil, nil); err == nil || !strings.Contains(err.Error(), "column mismatch") {
		t.Errorf("expected column mismatch error, got: %v", err)
	}

	// 5. Tensor data type mismatch
	tensorF16 := devCtx.NewTensorWithType(1, 4, device.DataTypeF16)
	if err := client.StreamEmbeddings(ctx, []*device.Tensor{tensorA, tensorF16}, nil, nil); err == nil || !strings.Contains(err.Error(), "data type mismatch") {
		t.Errorf("expected data type mismatch error, got: %v", err)
	}

	// 6. Normal streaming with F32 and ids fewer than rows (triggers AppendNull)
	err := client.StreamEmbeddings(ctx, []*device.Tensor{tensorA}, []string{}, map[string]string{"k": "v"})
	if err != nil {
		t.Fatalf("StreamEmbeddings with F32 failed: %v", err)
	}

	// 7. Normal streaming with F16
	err = client.StreamEmbeddings(ctx, []*device.Tensor{tensorF16}, []string{"f16_id"}, nil)
	if err != nil {
		t.Fatalf("StreamEmbeddings with F16 failed: %v", err)
	}
}

func createTestRecord(alloc memory.Allocator, ids []string, hasNullId bool, vectors [][]float32) arrow.Record {
	vecDim := len(vectors[0])
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "vector", Type: arrow.FixedSizeListOf(int32(vecDim), arrow.PrimitiveTypes.Float32)},
	}, nil)

	builder := array.NewRecordBuilder(alloc, schema)
	defer builder.Release()

	idBuilder := builder.Field(0).(*array.StringBuilder)
	vecListBuilder := builder.Field(1).(*array.FixedSizeListBuilder)
	vecValBuilder := vecListBuilder.ValueBuilder().(*array.Float32Builder)

	for i, v := range vectors {
		if hasNullId && i%2 == 1 {
			idBuilder.AppendNull()
		} else if i < len(ids) {
			idBuilder.Append(ids[i])
		} else {
			idBuilder.AppendNull()
		}

		vecListBuilder.Append(true)
		for _, val := range v {
			vecValBuilder.Append(val)
		}
	}
	return builder.NewRecord()
}

func createRecordNoVector(alloc memory.Allocator) arrow.Record {
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.BinaryTypes.String, Nullable: true},
	}, nil)
	builder := array.NewRecordBuilder(alloc, schema)
	defer builder.Release()
	builder.Field(0).(*array.StringBuilder).Append("only_id")
	return builder.NewRecord()
}

func TestFlightClient_DoGet_ServerInteractions(t *testing.T) {
	srv := &testFlightServer{hasEndpoint: true}
	client, cleanup := startTestFlightServer(t, srv)
	defer cleanup()

	ctx := context.Background()

	// 1. Client has no endpoints
	srv.mu.Lock()
	srv.hasEndpoint = false
	srv.mu.Unlock()
	if _, err := client.DoGet(ctx, []string{"doc1"}); err == nil || !strings.Contains(err.Error(), "no endpoints available") {
		t.Errorf("expected no endpoints available error, got: %v", err)
	}

	// 2. Client has endpoint, returns valid records with null and non-null IDs
	srv.mu.Lock()
	srv.hasEndpoint = true
	recWithData := createTestRecord(client.allocator, []string{"doc1", "doc2"}, true, [][]float32{{1.0, 2.0}, {3.0, 4.0}})
	srv.doGetRecords = []arrow.Record{recWithData}
	srv.mu.Unlock()

	batch, err := client.DoGet(ctx, []string{"doc1"})
	if err != nil {
		t.Fatalf("DoGet failed: %v", err)
	}
	if len(batch.Vectors) != 2 {
		t.Errorf("expected 2 vectors, got %d", len(batch.Vectors))
	}
	if len(batch.Ids) != 2 || batch.Ids[0] != "doc1" || batch.Ids[1] != "" {
		t.Errorf("unexpected batch IDs: %v", batch.Ids)
	}

	// 3. Record without vector column (vectorIdx < 0)
	srv.mu.Lock()
	recNoVec := createRecordNoVector(client.allocator)
	srv.doGetRecords = []arrow.Record{recNoVec}
	srv.mu.Unlock()
	batch, err = client.DoGet(ctx, []string{"doc1"})
	if err != nil {
		t.Fatalf("DoGet failed for record without vector: %v", err)
	}
	if len(batch.Vectors) != 0 {
		t.Errorf("expected 0 vectors, got %d", len(batch.Vectors))
	}

	// 4. Server stream error after sending records (triggers reader.Err())
	srv.mu.Lock()
	recWithData2 := createTestRecord(client.allocator, []string{"doc1"}, false, [][]float32{{1.0, 2.0}})
	srv.doGetRecords = []arrow.Record{recWithData2}
	srv.doGetErrAfterRecord = errors.New("simulated error after record")
	srv.mu.Unlock()
	if _, err := client.DoGet(ctx, []string{"doc1"}); err == nil || !strings.Contains(err.Error(), "error reading records") {
		t.Errorf("expected error reading records error, got: %v", err)
	}
}

func TestFlightClient_GetSchema_ServerInteractions(t *testing.T) {
	vecDim := 4
	targetSchema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "vector", Type: arrow.FixedSizeListOf(int32(vecDim), arrow.PrimitiveTypes.Float32)},
	}, nil)

	srv := &testFlightServer{schema: targetSchema}
	client, cleanup := startTestFlightServer(t, srv)
	defer cleanup()

	ctx := context.Background()

	// 1. Happy path: valid schema
	schema, err := client.GetSchema(ctx)
	if err != nil {
		t.Fatalf("GetSchema failed: %v", err)
	}
	if schema == nil || len(schema.Fields()) != 2 {
		t.Errorf("unexpected schema: %v", schema)
	}

	// 2. Invalid schema bytes returned
	srv.mu.Lock()
	srv.schema = nil // returns "invalid-schema" bytes
	srv.mu.Unlock()
	if _, err := client.GetSchema(ctx); err == nil || !strings.Contains(err.Error(), "failed to deserialize schema") {
		t.Errorf("expected failed to deserialize schema error, got: %v", err)
	}

	// 3. Deserialized schema is nil
	orig := flightDeserializeSchema
	defer func() { flightDeserializeSchema = orig }()
	flightDeserializeSchema = func(info []byte, mem memory.Allocator) (*arrow.Schema, error) {
		return nil, nil
	}
	if _, err := client.GetSchema(ctx); err == nil || !strings.Contains(err.Error(), "deserialized schema is nil") {
		t.Errorf("expected deserialized schema is nil error, got: %v", err)
	}
}

func TestFlightClient_GetFlightInfo_ServerInteractions(t *testing.T) {
	srv := &testFlightServer{hasEndpoint: true}
	client, cleanup := startTestFlightServer(t, srv)
	defer cleanup()

	ctx := context.Background()

	// 1. Happy path
	info, err := client.GetFlightInfo(ctx, "path1", "path2")
	if err != nil {
		t.Fatalf("GetFlightInfo failed: %v", err)
	}
	if info == nil || len(info.GetEndpoint()) == 0 {
		t.Errorf("unexpected flight info: %v", info)
	}

	// 2. Not connected
	unconnClient, _ := NewFlightClient("127.0.0.1", 3000, "127.0.0.1", 3001)
	if _, err := unconnClient.GetFlightInfo(ctx, "path1"); err == nil || !strings.Contains(err.Error(), "client not connected") {
		t.Errorf("expected client not connected error, got: %v", err)
	}
}

// Mock flight client and stream wrappers for error path testing
type mockFlightClientWrapper struct {
	flight.Client
	doPutFunc         func(ctx context.Context, opts ...grpc.CallOption) (flight.FlightService_DoPutClient, error)
	doGetFunc         func(ctx context.Context, in *flight.Ticket, opts ...grpc.CallOption) (flight.FlightService_DoGetClient, error)
	getFlightInfoFunc func(ctx context.Context, in *flight.FlightDescriptor, opts ...grpc.CallOption) (*flight.FlightInfo, error)
	getSchemaFunc     func(ctx context.Context, in *flight.FlightDescriptor, opts ...grpc.CallOption) (*flight.SchemaResult, error)
	closeFunc         func() error
}

func (m *mockFlightClientWrapper) DoPut(ctx context.Context, opts ...grpc.CallOption) (flight.FlightService_DoPutClient, error) {
	if m.doPutFunc != nil {
		return m.doPutFunc(ctx, opts...)
	}
	return nil, errors.New("mock DoPut stream error")
}

func (m *mockFlightClientWrapper) DoGet(ctx context.Context, in *flight.Ticket, opts ...grpc.CallOption) (flight.FlightService_DoGetClient, error) {
	if m.doGetFunc != nil {
		return m.doGetFunc(ctx, in, opts...)
	}
	return nil, errors.New("mock DoGet stream error")
}

func (m *mockFlightClientWrapper) GetFlightInfo(ctx context.Context, in *flight.FlightDescriptor, opts ...grpc.CallOption) (*flight.FlightInfo, error) {
	if m.getFlightInfoFunc != nil {
		return m.getFlightInfoFunc(ctx, in, opts...)
	}
	return nil, errors.New("mock GetFlightInfo error")
}

func (m *mockFlightClientWrapper) GetSchema(ctx context.Context, in *flight.FlightDescriptor, opts ...grpc.CallOption) (*flight.SchemaResult, error) {
	if m.getSchemaFunc != nil {
		return m.getSchemaFunc(ctx, in, opts...)
	}
	return nil, errors.New("mock GetSchema error")
}

func (m *mockFlightClientWrapper) Close() error {
	if m.closeFunc != nil {
		return m.closeFunc()
	}
	return nil
}

type mockDoPutClient struct {
	grpc.ClientStream
	sendFunc      func(*flight.FlightData) error
	closeSendFunc func() error
	recvFunc      func() (*flight.PutResult, error)
}

func (m *mockDoPutClient) Send(d *flight.FlightData) error {
	if m.sendFunc != nil {
		return m.sendFunc(d)
	}
	return nil
}

func (m *mockDoPutClient) CloseSend() error {
	if m.closeSendFunc != nil {
		return m.closeSendFunc()
	}
	return nil
}

func (m *mockDoPutClient) Recv() (*flight.PutResult, error) {
	if m.recvFunc != nil {
		return m.recvFunc()
	}
	return nil, io.EOF
}

func (m *mockDoPutClient) Header() (metadata.MD, error) { return nil, nil }
func (m *mockDoPutClient) Trailer() metadata.MD         { return nil }
func (m *mockDoPutClient) CloseAndRecv() (*flight.PutResult, error) {
	return m.Recv()
}
func (m *mockDoPutClient) SendMsg(m_ interface{}) error { return nil }
func (m *mockDoPutClient) RecvMsg(m_ interface{}) error { return nil }

type mockDoGetClient struct {
	grpc.ClientStream
	recvFunc func() (*flight.FlightData, error)
}

func (m *mockDoGetClient) Recv() (*flight.FlightData, error) {
	if m.recvFunc != nil {
		return m.recvFunc()
	}
	return nil, io.EOF
}

func (m *mockDoGetClient) Header() (metadata.MD, error) { return nil, nil }
func (m *mockDoGetClient) Trailer() metadata.MD         { return nil }
func (m *mockDoGetClient) CloseSend() error             { return nil }
func (m *mockDoGetClient) SendMsg(m_ interface{}) error { return nil }
func (m *mockDoGetClient) RecvMsg(m_ interface{}) error { return nil }

func TestFlightClient_StreamErrorBranches(t *testing.T) {
	ctx := context.Background()

	// 1. DoPut stream creation failure
	fc, _ := NewFlightClient("127.0.0.1", 3000, "127.0.0.1", 3001)
	mock := &mockFlightClientWrapper{
		doPutFunc: func(ctx context.Context, opts ...grpc.CallOption) (flight.FlightService_DoPutClient, error) {
			return nil, errors.New("doput failed")
		},
	}
	fc.client = mock

	vectors := [][]float32{{1.0, 2.0}}
	if err := fc.DoPut(ctx, vectors, nil, nil); err == nil || !strings.Contains(err.Error(), "failed to create DoPut stream") {
		t.Errorf("expected failed to create DoPut stream error, got: %v", err)
	}

	// 2. DoPut Write error (Send fails)
	fc.client = &mockFlightClientWrapper{
		doPutFunc: func(ctx context.Context, opts ...grpc.CallOption) (flight.FlightService_DoPutClient, error) {
			return &mockDoPutClient{
				sendFunc: func(d *flight.FlightData) error {
					return errors.New("send error")
				},
			}, nil
		},
	}
	if err := fc.DoPut(ctx, vectors, nil, nil); err == nil || !strings.Contains(err.Error(), "failed to write record") {
		t.Errorf("expected failed to write record error, got: %v", err)
	}

	// 3. DoPut CloseSend error
	fc.client = &mockFlightClientWrapper{
		doPutFunc: func(ctx context.Context, opts ...grpc.CallOption) (flight.FlightService_DoPutClient, error) {
			return &mockDoPutClient{
				closeSendFunc: func() error {
					return errors.New("close send error")
				},
			}, nil
		},
	}
	if err := fc.DoPut(ctx, vectors, nil, nil); err == nil || !strings.Contains(err.Error(), "failed to close send stream") {
		t.Errorf("expected failed to close send stream error, got: %v", err)
	}

	// 4. DoPut Recv error
	fc.client = &mockFlightClientWrapper{
		doPutFunc: func(ctx context.Context, opts ...grpc.CallOption) (flight.FlightService_DoPutClient, error) {
			return &mockDoPutClient{
				recvFunc: func() (*flight.PutResult, error) {
					return nil, errors.New("recv error")
				},
			}, nil
		},
	}
	if err := fc.DoPut(ctx, vectors, nil, nil); err == nil || !strings.Contains(err.Error(), "error receiving acknowledgment") {
		t.Errorf("expected error receiving acknowledgment error, got: %v", err)
	}

	// 5. DoPutTensor stream creation failure
	fc.client = &mockFlightClientWrapper{
		doPutFunc: func(ctx context.Context, opts ...grpc.CallOption) (flight.FlightService_DoPutClient, error) {
			return nil, errors.New("stream err")
		},
	}
	if _, err := fc.DoPutTensor(ctx, []float32{1.0}, []int32{1}, nil); err == nil || !strings.Contains(err.Error(), "failed to create DoPut stream") {
		t.Errorf("expected failed to create DoPut stream error, got: %v", err)
	}

	// 6. DoPutTensor Write error
	fc.client = &mockFlightClientWrapper{
		doPutFunc: func(ctx context.Context, opts ...grpc.CallOption) (flight.FlightService_DoPutClient, error) {
			return &mockDoPutClient{
				sendFunc: func(d *flight.FlightData) error {
					return errors.New("send tensor err")
				},
			}, nil
		},
	}
	if _, err := fc.DoPutTensor(ctx, []float32{1.0}, []int32{1}, nil); err == nil || !strings.Contains(err.Error(), "failed to write record") {
		t.Errorf("expected failed to write record error, got: %v", err)
	}

	// 7. DoPutTensor CloseSend error
	fc.client = &mockFlightClientWrapper{
		doPutFunc: func(ctx context.Context, opts ...grpc.CallOption) (flight.FlightService_DoPutClient, error) {
			return &mockDoPutClient{
				closeSendFunc: func() error {
					return errors.New("closesend err")
				},
			}, nil
		},
	}
	if _, err := fc.DoPutTensor(ctx, []float32{1.0}, []int32{1}, nil); err == nil || !strings.Contains(err.Error(), "failed to close send stream") {
		t.Errorf("expected failed to close send stream error, got: %v", err)
	}

	// 8. DoPutTensor Recv error
	fc.client = &mockFlightClientWrapper{
		doPutFunc: func(ctx context.Context, opts ...grpc.CallOption) (flight.FlightService_DoPutClient, error) {
			return &mockDoPutClient{
				recvFunc: func() (*flight.PutResult, error) {
					return nil, errors.New("recv tensor err")
				},
			}, nil
		},
	}
	if _, err := fc.DoPutTensor(ctx, []float32{1.0}, []int32{1}, nil); err == nil || !strings.Contains(err.Error(), "error receiving result") {
		t.Errorf("expected error receiving result error, got: %v", err)
	}

	// 9. StreamEmbeddings stream creation failure
	devCtx := device.NewContext()
	tensor := devCtx.NewTensorWithType(1, 2, device.DataTypeF32)
	fc.client = &mockFlightClientWrapper{
		doPutFunc: func(ctx context.Context, opts ...grpc.CallOption) (flight.FlightService_DoPutClient, error) {
			return nil, errors.New("stream err")
		},
	}
	if err := fc.StreamEmbeddings(ctx, []*device.Tensor{tensor}, nil, nil); err == nil || !strings.Contains(err.Error(), "failed to create DoPut stream") {
		t.Errorf("expected failed to create DoPut stream error, got: %v", err)
	}

	// 10. StreamEmbeddings Write error
	fc.client = &mockFlightClientWrapper{
		doPutFunc: func(ctx context.Context, opts ...grpc.CallOption) (flight.FlightService_DoPutClient, error) {
			return &mockDoPutClient{
				sendFunc: func(d *flight.FlightData) error {
					return errors.New("send stream err")
				},
			}, nil
		},
	}
	if err := fc.StreamEmbeddings(ctx, []*device.Tensor{tensor}, nil, nil); err == nil || !strings.Contains(err.Error(), "failed to write record") {
		t.Errorf("expected failed to write record error, got: %v", err)
	}

	// 10b. StreamEmbeddings ToArrowArray error
	fc.client = &mockFlightClientWrapper{
		doPutFunc: func(ctx context.Context, opts ...grpc.CallOption) (flight.FlightService_DoPutClient, error) {
			return &mockDoPutClient{}, nil
		},
	}
	emptyTensor := &device.Tensor{}
	if err := fc.StreamEmbeddings(ctx, []*device.Tensor{emptyTensor}, nil, nil); err == nil || !strings.Contains(err.Error(), "failed to convert tensor 0 to Arrow") {
		t.Errorf("expected failed to convert tensor 0 to Arrow error, got: %v", err)
	}

	// 11. StreamEmbeddings CloseSend error
	fc.client = &mockFlightClientWrapper{
		doPutFunc: func(ctx context.Context, opts ...grpc.CallOption) (flight.FlightService_DoPutClient, error) {
			return &mockDoPutClient{
				closeSendFunc: func() error {
					return errors.New("close send stream err")
				},
			}, nil
		},
	}
	if err := fc.StreamEmbeddings(ctx, []*device.Tensor{tensor}, nil, nil); err == nil || !strings.Contains(err.Error(), "failed to close flight stream") {
		t.Errorf("expected failed to close flight stream error, got: %v", err)
	}

	// 12. StreamEmbeddings Recv error
	fc.client = &mockFlightClientWrapper{
		doPutFunc: func(ctx context.Context, opts ...grpc.CallOption) (flight.FlightService_DoPutClient, error) {
			return &mockDoPutClient{
				recvFunc: func() (*flight.PutResult, error) {
					return nil, errors.New("recv stream err")
				},
			}, nil
		},
	}
	if err := fc.StreamEmbeddings(ctx, []*device.Tensor{tensor}, nil, nil); err == nil || !strings.Contains(err.Error(), "error receiving acknowledgment") {
		t.Errorf("expected error receiving acknowledgment error, got: %v", err)
	}

	// 13. DoGet - GetFlightInfo error
	fc.client = &mockFlightClientWrapper{
		getFlightInfoFunc: func(ctx context.Context, in *flight.FlightDescriptor, opts ...grpc.CallOption) (*flight.FlightInfo, error) {
			return nil, errors.New("flight info err")
		},
	}
	if _, err := fc.DoGet(ctx, []string{"id"}); err == nil || !strings.Contains(err.Error(), "failed to get flight info") {
		t.Errorf("expected failed to get flight info error, got: %v", err)
	}

	// 14. DoGet - DoGet stream error
	fc.client = &mockFlightClientWrapper{
		getFlightInfoFunc: func(ctx context.Context, in *flight.FlightDescriptor, opts ...grpc.CallOption) (*flight.FlightInfo, error) {
			return &flight.FlightInfo{Endpoint: []*flight.FlightEndpoint{{Ticket: &flight.Ticket{Ticket: []byte("t")}}}}, nil
		},
		doGetFunc: func(ctx context.Context, in *flight.Ticket, opts ...grpc.CallOption) (flight.FlightService_DoGetClient, error) {
			return nil, errors.New("do get stream err")
		},
	}
	if _, err := fc.DoGet(ctx, []string{"id"}); err == nil || !strings.Contains(err.Error(), "failed to create DoGet stream") {
		t.Errorf("expected failed to create DoGet stream error, got: %v", err)
	}

	// 15. DoGet - NewRecordReader error
	fc.client = &mockFlightClientWrapper{
		getFlightInfoFunc: func(ctx context.Context, in *flight.FlightDescriptor, opts ...grpc.CallOption) (*flight.FlightInfo, error) {
			return &flight.FlightInfo{Endpoint: []*flight.FlightEndpoint{{Ticket: &flight.Ticket{Ticket: []byte("t")}}}}, nil
		},
		doGetFunc: func(ctx context.Context, in *flight.Ticket, opts ...grpc.CallOption) (flight.FlightService_DoGetClient, error) {
			return &mockDoGetClient{
				recvFunc: func() (*flight.FlightData, error) {
					return nil, errors.New("reader recv err")
				},
			}, nil
		},
	}
	if _, err := fc.DoGet(ctx, []string{"id"}); err == nil || !strings.Contains(err.Error(), "failed to create record reader") {
		t.Errorf("expected failed to create record reader error, got: %v", err)
	}

	// 16. GetSchema - GetSchema error
	fc.client = &mockFlightClientWrapper{
		getSchemaFunc: func(ctx context.Context, in *flight.FlightDescriptor, opts ...grpc.CallOption) (*flight.SchemaResult, error) {
			return nil, errors.New("schema rpc err")
		},
	}
	if _, err := fc.GetSchema(ctx); err == nil || !strings.Contains(err.Error(), "failed to get schema") {
		t.Errorf("expected failed to get schema error, got: %v", err)
	}
}
