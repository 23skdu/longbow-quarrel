package api

import (
	"context"
	"net"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/flight"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// The tests in this file talk to the Flight server with the Arrow client
// from the Arrow library itself.
//
// The repository used to carry its own Flight *client* (internal/arrow_client)
// that no production code called: the server half was live and started by
// the CLI, and the client half was only reachable from the tests. The
// client is gone, so the server is exercised the way a real client would
// exercise it.

// startFlightTestServer starts the inference Flight server on a free port
// and returns its address.
func startFlightTestServer(t *testing.T) string {
	t.Helper()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}

	server := NewInferenceFlightServer(lis.Addr().String(), &mockEngine{}, &tokenizer.Tokenizer{})
	go func() { _ = server.ServeListener(lis) }()

	// The server needs a moment to bind.
	time.Sleep(200 * time.Millisecond)

	return lis.Addr().String()
}

// newFlightTestClient connects to a Flight server with the Arrow client.
func newFlightTestClient(t *testing.T, addr string) flight.Client {
	t.Helper()

	client, err := flight.NewClientWithMiddleware(addr, nil, nil,
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("failed to create the flight client: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client
}

// TestFlightServerStreamsTheTokensTheClientReads checks that the token
// stream carries the columns a client reads.
//
// The server used to send "token_id", "logits" and "text" only, and the
// client looked for "id" and "vector" and ignored everything else: the
// remote worker came back with no tokens and no error.
func TestFlightServerStreamsTheTokensTheClientReads(t *testing.T) {
	addr := startFlightTestServer(t)
	client := newFlightTestClient(t, addr)

	stream, err := client.DoGet(context.Background(), &flight.Ticket{Ticket: []byte("prompt")})
	if err != nil {
		t.Fatalf("failed to start the DoGet stream: %v", err)
	}

	reader, err := flight.NewRecordReader(stream)
	if err != nil {
		t.Fatalf("failed to create the record reader: %v", err)
	}
	defer reader.Release()

	if !reader.Next() {
		t.Fatal("the server sent no record")
	}
	record := reader.RecordBatch()
	if record == nil {
		t.Fatal("the server sent an empty record")
	}
	if record.NumRows() == 0 {
		t.Fatal("the server sent a record with no rows")
	}

	names := make(map[string]bool)
	for _, field := range record.Schema().Fields() {
		names[field.Name] = true
	}
	for _, want := range []string{"id", "vector", "token_id"} {
		if !names[want] {
			t.Errorf("the stream has no %q column, so a client cannot read it", want)
		}
	}

	// The "vector" column carries the token, and the "id" column its
	// decimal form. The mock engine answers with the tokens 1, 2 and 3.
	vectorColumn := record.Column(record.Schema().FieldIndices("vector")[0])
	vectors, ok := vectorColumn.(*array.FixedSizeList)
	if !ok {
		t.Fatalf("the vector column is a %T, want a fixed size list", vectorColumn)
	}
	values, ok := vectors.ListValues().(*array.Float32)
	if !ok {
		t.Fatalf("the vector values are a %T, want float32", vectors.ListValues())
	}
	if values.Len() != 1 {
		t.Fatalf("the vector holds %d values, want 1", values.Len())
	}
	if got := values.Value(0); got != 1 {
		t.Errorf("the vector holds %v, want the mock engine's first token 1", got)
	}

	idColumn := record.Column(record.Schema().FieldIndices("id")[0])
	ids, ok := idColumn.(*array.String)
	if !ok {
		t.Fatalf("the id column is a %T, want a string", idColumn)
	}
	if got := ids.Value(0); got != "1" {
		t.Errorf("the id column holds %q, want %q", got, "1")
	}
}

// TestFlightServerSchemaIsReadable checks that the schema the server
// advertises is the schema its stream carries.
func TestFlightServerSchemaIsReadable(t *testing.T) {
	addr := startFlightTestServer(t)
	client := newFlightTestClient(t, addr)

	result, err := client.GetSchema(context.Background(), &flight.FlightDescriptor{
		Type: flight.DescriptorPATH,
		Path: []string{"generation_stream"},
	})
	if err != nil {
		t.Fatalf("GetSchema failed: %v", err)
	}

	schema, err := flight.DeserializeSchema(result.Schema, memory.NewGoAllocator())
	if err != nil {
		t.Fatalf("failed to read the schema: %v", err)
	}

	names := make(map[string]bool)
	for _, field := range schema.Fields() {
		names[field.Name] = true
	}
	for _, want := range []string{"id", "vector"} {
		if !names[want] {
			t.Errorf("the schema has no %q column, so a client cannot read it", want)
		}
	}

	if !schema.HasMetadata() {
		t.Fatal("the schema has no metadata")
	}
	meta := schema.Metadata()
	found := false
	for i := 0; i < meta.Len(); i++ {
		if meta.Keys()[i] == "QUARREL:device_id" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected QUARREL:device_id in the schema metadata")
	}
}

// TestFlightServerFlightInfoListsTheEndpoint checks the discovery call.
func TestFlightServerFlightInfoListsTheEndpoint(t *testing.T) {
	addr := startFlightTestServer(t)
	client := newFlightTestClient(t, addr)

	info, err := client.GetFlightInfo(context.Background(), &flight.FlightDescriptor{
		Type: flight.DescriptorPATH,
		Path: []string{"generation_stream"},
	})
	if err != nil {
		t.Fatalf("GetFlightInfo failed: %v", err)
	}
	if len(info.Endpoint) == 0 {
		t.Fatal("the server advertises no endpoint")
	}
	if got := string(info.Endpoint[0].Ticket.Ticket); got != "generation_stream" {
		t.Errorf("the endpoint ticket is %q, want %q", got, "generation_stream")
	}
}
