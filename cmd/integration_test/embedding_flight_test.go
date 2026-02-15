package main

import (
	"context"
	"flag"
	"fmt"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/arrow_client"
)

var (
	testDimension = flag.Int("dim", 384, "Embedding dimension for tests")
	testBatchSize = flag.Int("batch", 10, "Number of embeddings to test")
)

func TestEmbeddingFlightPipeline(t *testing.T) {
	flag.Parse()

	ctx := context.Background()

	client := arrow_client.NewMockFlightClient()
	defer func() { _ = client.Close() }()

	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Failed to connect: %v", err)
	}

	testVectors := make([][]float32, *testBatchSize)
	testIds := make([]string, *testBatchSize)

	for i := 0; i < *testBatchSize; i++ {
		testVectors[i] = make([]float32, *testDimension)
		testIds[i] = fmt.Sprintf("test_doc_%d", i)
		for j := 0; j < *testDimension; j++ {
			testVectors[i][j] = float32(i*j) / float32(*testDimension)
		}
	}

	metadata := map[string]string{
		"model":   "nomic-embed-text",
		"version": "1.5",
	}

	err := client.DoPut(ctx, testVectors, testIds, metadata)
	if err != nil {
		t.Fatalf("DoPut failed: %v", err)
	}

	t.Logf("Successfully sent %d embeddings", *testBatchSize)

	result, err := client.DoGet(ctx, testIds)
	if err != nil {
		t.Fatalf("DoGet failed: %v", err)
	}

	if len(result.Vectors) != *testBatchSize {
		t.Errorf("Expected %d vectors, got %d", *testBatchSize, len(result.Vectors))
	}

	for i, vec := range result.Vectors {
		if len(vec) != *testDimension {
			t.Errorf("Vector %d has dimension %d, expected %d", i, len(vec), *testDimension)
		}

		for j, val := range vec {
			if val != testVectors[i][j] {
				t.Errorf("Vector %d element %d mismatch: got %f, expected %f", i, j, val, testVectors[i][j])
			}
		}
	}

	t.Logf("Successfully retrieved and verified %d embeddings", len(result.Vectors))
}

func TestFlightClientDoPut(t *testing.T) {
	flag.Parse()

	ctx := context.Background()
	client := arrow_client.NewMockFlightClient()
	defer func() { _ = client.Close() }()

	_ = client.Connect(ctx)

	vectors := [][]float32{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
	}
	ids := []string{"doc1", "doc2"}

	err := client.DoPut(ctx, vectors, ids, nil)
	if err != nil {
		t.Errorf("DoPut failed: %v", err)
	}

	stored := client.GetStoredData()
	if len(stored) != 1 {
		t.Errorf("Expected 1 stored batch, got %d", len(stored))
	}

	if batch, ok := stored["doc1"]; ok {
		if len(batch.Vectors) != 2 {
			t.Errorf("Expected 2 vectors, got %d", len(batch.Vectors))
		}
	}
}

func TestFlightClientDoGet(t *testing.T) {
	ctx := context.Background()
	client := arrow_client.NewMockFlightClient()
	defer func() { _ = client.Close() }()

	_ = client.Connect(ctx)

	vectors := [][]float32{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
	}
	ids := []string{"doc1", "doc2"}

	_ = client.DoPut(ctx, vectors, ids, nil)

	result, err := client.DoGet(ctx, []string{"doc1"})
	if err != nil {
		t.Errorf("DoGet failed: %v", err)
	}

	if len(result.Vectors) != 2 {
		t.Errorf("Expected 2 vectors, got %d", len(result.Vectors))
	}
}

func TestFlightClientGetFlightInfo(t *testing.T) {
	ctx := context.Background()
	client := arrow_client.NewMockFlightClient()
	defer func() { _ = client.Close() }()

	_ = client.Connect(ctx)

	info, err := client.GetFlightInfo(ctx, "embeddings")
	if err != nil {
		t.Errorf("GetFlightInfo failed: %v", err)
	}

	if info == nil {
		t.Error("Expected non-nil FlightInfo")
	}
}

func TestRecordBatchCreation(t *testing.T) {
	vectors := [][]float32{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
	}
	ids := []string{"a", "b"}

	batch := arrow_client.NewRecordBatch(vectors, ids)

	if len(batch.Vectors) != 2 {
		t.Errorf("Expected 2 vectors, got %d", len(batch.Vectors))
	}

	if len(batch.Ids) != 2 {
		t.Errorf("Expected 2 IDs, got %d", len(batch.Ids))
	}

	if batch.Metadata == nil {
		t.Error("Expected non-nil metadata")
	}
}

func TestEmptyVectors(t *testing.T) {
	ctx := context.Background()
	client := arrow_client.NewMockFlightClient()
	defer func() { _ = client.Close() }()

	_ = client.Connect(ctx)

	err := client.DoPut(ctx, [][]float32{}, []string{}, nil)
	if err != nil {
		t.Logf("Mock client rejected empty vectors: %v", err)
	} else {
		stored := client.GetStoredData()
		t.Logf("Mock client accepted empty vectors, stored: %v", stored)
	}
}

func TestVectorRoundTrip(t *testing.T) {
	ctx := context.Background()
	client := arrow_client.NewMockFlightClient()
	defer func() { _ = client.Close() }()

	_ = client.Connect(ctx)

	original := [][]float32{
		{0.1, 0.2, 0.3, 0.4},
		{0.5, 0.6, 0.7, 0.8},
	}
	ids := []string{"round_trip_test"}

	_ = client.DoPut(ctx, original, ids, nil)

	result, err := client.DoGet(ctx, ids)
	if err != nil {
		t.Fatalf("DoGet failed: %v", err)
	}

	for i, vec := range result.Vectors {
		for j, val := range vec {
			if val != original[i][j] {
				t.Errorf("Round trip mismatch at [%d][%d]: got %f, expected %f", i, j, val, original[i][j])
			}
		}
	}
}

func TestMetadataPreservation(t *testing.T) {
	ctx := context.Background()
	client := arrow_client.NewMockFlightClient()
	defer func() { _ = client.Close() }()

	_ = client.Connect(ctx)

	vectors := [][]float32{{1.0, 2.0}}
	ids := []string{"meta_test"}
	metadata := map[string]string{
		"source":  "test",
		"version": "1.0",
	}

	_ = client.DoPut(ctx, vectors, ids, metadata)

	stored := client.GetStoredData()
	batch := stored["meta_test"]

	if batch.Metadata["source"] != "test" {
		t.Errorf("Metadata not preserved: got %s", batch.Metadata["source"])
	}

	if batch.Metadata["version"] != "1.0" {
		t.Errorf("Metadata not preserved: got %s", batch.Metadata["version"])
	}
}

func main() {
	testing.Main(func(pat, str string) (bool, error) { return true, nil },
		[]testing.InternalTest{
			{Name: "TestEmbeddingFlightPipeline", F: TestEmbeddingFlightPipeline},
			{Name: "TestFlightClientDoPut", F: TestFlightClientDoPut},
			{Name: "TestFlightClientDoGet", F: TestFlightClientDoGet},
			{Name: "TestFlightClientGetFlightInfo", F: TestFlightClientGetFlightInfo},
			{Name: "TestRecordBatchCreation", F: TestRecordBatchCreation},
			{Name: "TestEmptyVectors", F: TestEmptyVectors},
			{Name: "TestVectorRoundTrip", F: TestVectorRoundTrip},
			{Name: "TestMetadataPreservation", F: TestMetadataPreservation},
		},
		nil, nil)
}
