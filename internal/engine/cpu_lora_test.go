package engine

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// buildLoRAFile writes a LoRA adapter in the GGUF format.
//
// The adapter has one layer with an A matrix of [dimIn, rank] and a B
// matrix of [rank, dimOut], so the delta it produces is a known value.
func buildLoRAFile(t *testing.T, dir, name string, dimIn, rank, dimOut int, alpha float32) string {
	t.Helper()

	aData := make([]float32, dimIn*rank)
	for i := range aData {
		aData[i] = float32(i%5) / 10.0
	}
	bData := make([]float32, rank*dimOut)
	for i := range bData {
		bData[i] = float32(i%3) / 20.0
	}

	toF32 := func(values []float32) []byte {
		out := make([]byte, len(values)*4)
		for i, v := range values {
			bits := math.Float32bits(v)
			out[i*4] = byte(bits)
			out[i*4+1] = byte(bits >> 8)
			out[i*4+2] = byte(bits >> 16)
			out[i*4+3] = byte(bits >> 24)
		}
		return out
	}

	kv := map[string]interface{}{
		"adapter.lora_alpha": alpha,
		// The tokenizer is needed by the model swap.
		"tokenizer.ggml.tokens":     []string{"<unk>", "a", "b", "c"},
		"tokenizer.ggml.scores":     []float32{0, -1, -2, -3},
		"tokenizer.ggml.token_type": []int32{2, 1, 1, 1},
	}
	tensors := []gguf.TensorData{
		{
			// The GGUF dimensions are the row length first, so a matrix
			// of the logical shape [dimIn, rank] is stored as
			// [rank, dimIn].
			Name:       "blk.0.attn_q.lora_A.weight",
			Dimensions: []uint64{uint64(rank), uint64(dimIn)},
			Type:       gguf.GGMLTypeF32,
			Data:       toF32(aData),
		},
		{
			Name:       "blk.0.attn_q.lora_B.weight",
			Dimensions: []uint64{uint64(dimOut), uint64(rank)},
			Type:       gguf.GGMLTypeF32,
			Data:       toF32(bData),
		},
	}

	path := filepath.Join(dir, name)
	if err := gguf.WriteFile(path, kv, tensors, 32); err != nil {
		t.Fatalf("failed to write the adapter: %v", err)
	}
	return path
}

// TestCPUEngineLoadAdapter covers the CPU engine's LoRA loading and
// merging.
//
// The path had no test at all, and it is the one the REST API's
// "/v1/adapters/load" endpoint and the CLI's -lora flag use.
func TestCPUEngineLoadAdapter(t *testing.T) {
	const (
		dimIn  = 4
		rank   = 2
		dimOut = 3
		alpha  = 8.0
	)

	dir := t.TempDir()
	path := buildLoRAFile(t, dir, "adapter.gguf", dimIn, rank, dimOut, alpha)

	// The target weight is [dimOut, dimIn], which is the row-major shape
	// of a linear layer.
	weights := &CPUWeights{
		AttnQ: [][]float32{make([]float32, dimOut*dimIn)},
	}
	cfg := config.Default()
	cfg.Layers = 1

	e := &CPUEngine{weights: weights, config: cfg}

	if err := e.LoadAdapter(path, "adapter-1"); err != nil {
		t.Fatalf("LoadAdapter failed: %v", err)
	}

	// The adapter is recorded.
	adapters := e.ListAdapters()
	if len(adapters) != 1 {
		t.Fatalf("the engine has %d adapters, want 1", len(adapters))
	}
	if adapters[0].ID != "adapter-1" {
		t.Errorf("the adapter is %q, want adapter-1", adapters[0].ID)
	}

	// The delta was merged into the target weight. The expected value
	// is sum_r(A[i][r] * B[r][o]) * alpha / rank.
	aData := make([]float32, dimIn*rank)
	for i := range aData {
		aData[i] = float32(i%5) / 10.0
	}
	bData := make([]float32, rank*dimOut)
	for i := range bData {
		bData[i] = float32(i%3) / 20.0
	}
	scale := alpha / float32(rank)

	merged := weights.AttnQ[0]
	for o := 0; o < dimOut; o++ {
		for i := 0; i < dimIn; i++ {
			var sum float32
			for r := 0; r < rank; r++ {
				sum += aData[i*rank+r] * bData[r*dimOut+o]
			}
			want := sum * scale
			if math.Abs(float64(merged[o*dimIn+i]-want)) > 1e-6 {
				t.Errorf("the merged weight %d is %v, want %v", o*dimIn+i, merged[o*dimIn+i], want)
			}
		}
	}
}

// TestCPUEngineLoadAdapterErrors covers the error paths of the LoRA
// loading.
func TestCPUEngineLoadAdapterErrors(t *testing.T) {
	e := &CPUEngine{weights: &CPUWeights{}}

	if err := e.LoadAdapter("/nonexistent/adapter.gguf", "adapter-1"); err == nil {
		t.Error("expected an error for a missing adapter file")
	}
}

// TestCPUEngineLoadAdapterIgnoresUnknownLayers checks that an adapter for
// a layer the engine does not have is ignored rather than merged into the
// wrong weights.
func TestCPUEngineLoadAdapterIgnoresUnknownLayers(t *testing.T) {
	dir := t.TempDir()
	path := buildLoRAFile(t, dir, "adapter.gguf", 4, 2, 3, 8.0)

	// The engine has no layers, so nothing may be merged.
	weights := &CPUWeights{
		AttnQ: [][]float32{make([]float32, 12)},
	}
	cfg := config.Default()
	cfg.Layers = 0

	e := &CPUEngine{weights: weights, config: cfg}
	if err := e.LoadAdapter(path, "adapter-1"); err != nil {
		t.Fatalf("LoadAdapter failed: %v", err)
	}

	for i, v := range weights.AttnQ[0] {
		if v != 0 {
			t.Errorf("the weight %d is %v, want it untouched", i, v)
		}
	}
}

// TestCPUEngineListAdaptersIsSorted checks the adapter listing.
func TestCPUEngineListAdaptersIsSorted(t *testing.T) {
	e := &CPUEngine{weights: &CPUWeights{}}
	e.adapters = []AdapterInfo{
		{ID: "b"},
		{ID: "a"},
		{ID: "c"},
	}

	adapters := e.ListAdapters()
	if len(adapters) != 3 {
		t.Fatalf("the engine has %d adapters, want 3", len(adapters))
	}
	if adapters[0].ID != "a" || adapters[1].ID != "b" || adapters[2].ID != "c" {
		t.Errorf("the adapters are not sorted: %v", adapters)
	}
}

// TestCPUEngineSwapModel covers the model swap's error path and the
// successful reload.
func TestCPUEngineSwapModel(t *testing.T) {
	dir := t.TempDir()
	// The swap target has to be a model, not an adapter: SwapModel validates
	// the tensor table and rejects a file with no embedding table.
	path := buildTestGGUF(t, dir)

	e := &CPUEngine{weights: &CPUWeights{}}
	if err := e.SwapModel(path, config.Default()); err != nil {
		t.Errorf("SwapModel failed: %v", err)
	}
}

// TestDequantizeRow covers the row dequantizer, which the quantized
// weight path uses.
func TestDequantizeRow(t *testing.T) {
	// A float32 tensor.
	data := make([]float32, 64)
	for i := range data {
		data[i] = float32(i) / 8.0
	}
	raw := make([]byte, len(data)*4)
	for i, v := range data {
		bits := math.Float32bits(v)
		raw[i*4] = byte(bits)
		raw[i*4+1] = byte(bits >> 8)
		raw[i*4+2] = byte(bits >> 16)
		raw[i*4+3] = byte(bits >> 24)
	}

	tensor := &gguf.TensorInfo{
		Name:       "w",
		Dimensions: []uint64{4, 16},
		Type:       gguf.GGMLTypeF32,
		Data:       raw,
	}

	row := dequantizeRow(tensor, 1, 16)
	if len(row) != 16 {
		t.Fatalf("the row has %d values, want 16", len(row))
	}
	for i := range row {
		want := data[16+i]
		if math.Abs(float64(row[i]-want)) > 1e-6 {
			t.Errorf("the value %d is %v, want %v", i, row[i], want)
		}
	}

	// A row past the end of the data is empty rather than a panic.
	past := dequantizeRow(tensor, 100, 16)
	if len(past) != 16 {
		t.Errorf("a row past the end has %d values, want 16", len(past))
	}
	for i, v := range past {
		if v != 0 {
			t.Errorf("the value %d is %v, want 0", i, v)
		}
	}

	// An unsupported type is empty.
	unsupported := dequantizeRow(&gguf.TensorInfo{
		Name:       "w",
		Dimensions: []uint64{4},
		Type:       gguf.GGMLTypeMXFP4,
		Data:       raw,
	}, 0, 16)
	if len(unsupported) != 16 {
		t.Errorf("an unsupported type has %d values, want 16", len(unsupported))
	}

	// A nil tensor is empty.
	if got := dequantizeRow(nil, 0, 8); len(got) != 8 {
		t.Errorf("a nil tensor has %d values, want 8", len(got))
	}
}

// TestCPUWeightsMatVecQuantized covers the quantized weight path of the
// matrix-vector product.
//
// The Q4_K path is the one the engine uses for a quantized model, and it
// is the path the CPU and the GPU disagreed about.
func TestCPUWeightsMatVecQuantized(t *testing.T) {
	const cols = 256
	const rows = 2

	weights := make([]float32, rows*cols)
	for i := range weights {
		weights[i] = float32((i*29)%61-30) / 30.0
	}

	quantized, err := gguf.QuantizeWeightsToQ4K(weights, len(weights))
	if err != nil {
		t.Fatalf("failed to quantize: %v", err)
	}

	tensor := &gguf.TensorInfo{
		Name: "blk.0.attn_q.weight",
		// The GGUF dimensions are the row length first, so a matrix of
		// the logical shape [rows, cols] is stored as [cols, rows].
		Dimensions: []uint64{uint64(cols), uint64(rows)},
		Type:       gguf.GGMLTypeQ4_K,
		Data:       quantized,
	}

	w := &CPUWeights{}
	vector := make([]float32, cols)
	for i := range vector {
		vector[i] = float32(i%7) / 7.0
	}

	got := w.MatVec(nil, tensor, vector)
	if len(got) != rows {
		t.Fatalf("the product has %d values, want %d", len(got), rows)
	}

	// The result is close to the float32 product.
	for r := 0; r < rows; r++ {
		var want float32
		for c := 0; c < cols; c++ {
			want += weights[r*cols+c] * vector[c]
		}
		if math.Abs(float64(got[r]-want)) > 5.0 {
			t.Errorf("the row %d is %v, want about %v", r, got[r], want)
		}
	}
}

// TestCPUWeightsMatVecF32 covers the float32 weight path.
func TestCPUWeightsMatVecF32(t *testing.T) {
	const cols = 4
	const rows = 2

	weights := make([]float32, rows*cols)
	for i := range weights {
		weights[i] = float32(i)
	}
	raw := make([]byte, len(weights)*4)
	for i, v := range weights {
		bits := math.Float32bits(v)
		raw[i*4] = byte(bits)
		raw[i*4+1] = byte(bits >> 8)
		raw[i*4+2] = byte(bits >> 16)
		raw[i*4+3] = byte(bits >> 24)
	}

	tensor := &gguf.TensorInfo{
		Name:       "w",
		Dimensions: []uint64{uint64(cols), uint64(rows)},
		Type:       gguf.GGMLTypeF32,
		Data:       raw,
	}

	w := &CPUWeights{}
	vector := []float32{1, 1, 1, 1}
	got := w.MatVec(nil, tensor, vector)
	if len(got) != rows {
		t.Fatalf("the product has %d values, want %d", len(got), rows)
	}
	if got[0] != 6 {
		t.Errorf("the first row is %v, want 6", got[0])
	}
	if got[1] != 22 {
		t.Errorf("the second row is %v, want 22", got[1])
	}
}

// TestCPUEngineSwapModelMissingFile checks the guard.
func TestCPUEngineSwapModelMissingFile(t *testing.T) {
	e := &CPUEngine{weights: &CPUWeights{}}
	if err := e.SwapModel(filepath.Join(t.TempDir(), "missing.gguf"), config.Default()); err == nil {
		t.Error("expected an error for a missing model")
	}
	_ = os.Remove(filepath.Join(t.TempDir(), "missing.gguf"))
}
