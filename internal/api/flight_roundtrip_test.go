package api

import (
	"context"
	"net"
	"strconv"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/arrow_client"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

// TestFlightInferenceRoundTrip checks that the Arrow Flight server and the
// Arrow Flight client agree on the wire format.
//
// They did not. The server sent the generated tokens in the columns
// "token_id", "logits" and "text", and the client looked for "id" and
// "vector" and ignored everything else. The client therefore read a
// stream with no vectors in it, and the remote worker came back with **no
// tokens and no error**: the whole single-request distributed inference
// path was broken.
func TestFlightInferenceRoundTrip(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}

	server := NewInferenceFlightServer(lis.Addr().String(), &engine.MockEngine{}, &tokenizer.Tokenizer{})
	go func() { _ = server.ServeListener(lis) }()
	time.Sleep(200 * time.Millisecond)

	host, portStr, err := net.SplitHostPort(lis.Addr().String())
	if err != nil {
		t.Fatalf("failed to split the host and port: %v", err)
	}
	port, _ := strconv.Atoi(portStr)

	client, err := arrow_client.NewFlightClient(host, port, host, port)
	if err != nil {
		t.Fatalf("failed to create the client: %v", err)
	}
	if err := client.Connect(context.Background()); err != nil {
		t.Fatalf("failed to connect: %v", err)
	}
	defer func() { _ = client.Close() }()

	batch, err := client.DoGet(context.Background(), []string{"prompt"})
	if err != nil {
		t.Fatalf("DoGet failed: %v", err)
	}
	if len(batch.Vectors) == 0 {
		t.Fatal("the client read no vectors from the server")
	}
	for i, vec := range batch.Vectors {
		if len(vec) == 0 {
			t.Errorf("vector %d is empty", i)
			continue
		}
		if vec[0] != 42 {
			t.Errorf("vector %d starts with %v, want the mock engine's token 42", i, vec[0])
		}
	}
}

// TestFlightSchemaIsReadable checks that the server's schema and the
// client's expectations line up, by asking the server for its schema and
// looking for the columns the client reads.
func TestFlightSchemaIsReadable(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}

	server := NewInferenceFlightServer(lis.Addr().String(), &engine.MockEngine{}, &tokenizer.Tokenizer{})
	go func() { _ = server.ServeListener(lis) }()
	time.Sleep(200 * time.Millisecond)

	host, portStr, err := net.SplitHostPort(lis.Addr().String())
	if err != nil {
		t.Fatalf("failed to split the host and port: %v", err)
	}
	port, _ := strconv.Atoi(portStr)

	client, err := arrow_client.NewFlightClient(host, port, host, port)
	if err != nil {
		t.Fatalf("failed to create the client: %v", err)
	}
	if err := client.Connect(context.Background()); err != nil {
		t.Fatalf("failed to connect: %v", err)
	}
	defer func() { _ = client.Close() }()

	schema, err := client.GetSchema(context.Background())
	if err != nil {
		t.Fatalf("GetSchema failed: %v", err)
	}

	names := make(map[string]bool)
	for _, field := range schema.Fields() {
		names[field.Name] = true
	}
	for _, want := range []string{"id", "vector"} {
		if !names[want] {
			t.Errorf("the schema has no %q column, so the client cannot read it", want)
		}
	}
}
