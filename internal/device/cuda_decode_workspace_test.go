//go:build linux && amd64 && cuda && cgo

package device

import (
	"math"
	"testing"
)

// TestCUDA_DecodeWorkspace_StableAddresses checks that the buffers of a
// decode workspace keep their addresses.
//
// A captured CUDA graph bakes in the address of every buffer it touches,
// so a workspace whose addresses move between calls would replay into
// whatever the allocator handed out last time.
func TestCUDA_DecodeWorkspace_StableAddresses(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	shape := DecodeWorkspaceShape{
		NumTokens: 4,
		Dim:       64,
		HiddenDim: 128,
		Heads:     4,
		KVHeads:   2,
		HeadDim:   16,
		BatchSize: 2,
		MaxBlocks: 16,
		VocabSize: 128,
	}

	workspace, err := NewDecodeWorkspace(ctx, shape)
	if err != nil {
		t.Fatalf("failed to build the workspace: %v", err)
	}
	defer workspace.Free()

	first := workspace.Addresses()
	if len(first) == 0 {
		t.Fatal("the workspace holds no buffers")
	}
	for name, addr := range first {
		if addr == 0 {
			t.Errorf("the buffer %s has no address", name)
		}
	}

	// Two calls for the same shape give the same addresses: the buffers
	// are allocated once and reused.
	secondAddresses := workspace.Addresses()
	for name, addr := range first {
		if secondAddresses[name] != addr {
			t.Errorf("the buffer %s moved: %x then %x", name, addr, secondAddresses[name])
		}
	}

	// A second workspace of the same shape is a different set of
	// buffers, and the addresses of the first one are still the ones a
	// captured graph holds.
	other, err := NewDecodeWorkspace(ctx, shape)
	if err != nil {
		t.Fatalf("failed to build the second workspace: %v", err)
	}
	defer other.Free()

	thirdAddresses := workspace.Addresses()
	for name, addr := range first {
		if thirdAddresses[name] != addr {
			t.Errorf("the buffer %s moved after another workspace was built: %x then %x",
				name, addr, thirdAddresses[name])
		}
	}

	// A different shape is not compatible.
	if workspace.Compatible(DecodeWorkspaceShape{
		NumTokens: 8,
		Dim:       64,
		HiddenDim: 128,
		BatchSize: 2,
		MaxBlocks: 16,
		VocabSize: 128,
	}) {
		t.Error("a workspace of four tokens accepted a shape of eight")
	}
	if !workspace.Compatible(shape) {
		t.Error("the workspace rejected its own shape")
	}
}

// TestCUDA_GraphCaptureReplayMatchesTheEagerPath checks that a captured
// graph produces the same numbers as the eager path.
//
// The capture is only useful if the replay is the same computation: the
// graph holds the addresses of the buffers and the parameters of the
// kernels, so a capture that lost a step, or replayed a buffer that had
// been freed, would show up here.
func TestCUDA_GraphCaptureReplayMatchesTheEagerPath(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	const (
		rows = 8
		cols = 64
	)

	shape := DecodeWorkspaceShape{
		NumTokens: rows,
		Dim:       cols,
		HiddenDim: cols,
		BatchSize: 1,
		MaxBlocks: 4,
		VocabSize: 32,
	}
	workspace, err := NewDecodeWorkspace(ctx, shape)
	if err != nil {
		t.Fatalf("failed to build the workspace: %v", err)
	}
	defer workspace.Free()

	// The normalisation weight and the bias of the two kernels the
	// capture runs. Both kernels write into a tensor the caller names,
	// which is what makes them capturable: the output of a pooled
	// allocation can change between the capture and the replay.
	weight := ctx.NewTensorFP32(1, cols)
	if weight == nil {
		t.Fatal("failed to create the weight")
	}
	defer weight.Free()
	weightData := make([]float32, cols)
	for i := range weightData {
		weightData[i] = 1.0 + float32(i%13)*0.01
	}
	if err := weight.LoadFrom(weightData); err != nil {
		t.Fatalf("failed to load the weight: %v", err)
	}

	bias := ctx.NewTensorFP32(rows, cols)
	if bias == nil {
		t.Fatal("failed to create the bias")
	}
	defer bias.Free()
	biasData := make([]float32, rows*cols)
	for i := range biasData {
		biasData[i] = float32(i%11) * 0.02
	}
	if err := bias.LoadFrom(biasData); err != nil {
		t.Fatalf("failed to load the bias: %v", err)
	}

	input := ctx.NewTensorFP32(rows, cols)
	if input == nil {
		t.Fatal("failed to create the input")
	}
	defer input.Free()
	inputData := make([]float32, rows*cols)
	for i := range inputData {
		inputData[i] = float32(i%7) * 0.5
	}
	if err := input.LoadFrom(inputData); err != nil {
		t.Fatalf("failed to load the input: %v", err)
	}

	// The eager path: normalise, multiply, add.
	eager := ctx.NewTensorFP32(rows, cols)
	if eager == nil {
		t.Fatal("failed to create the eager output")
	}
	defer eager.Free()

	eagerNormed := ctx.NewTensorFP32(rows, cols)
	if eagerNormed == nil {
		t.Fatal("failed to create the eager normalisation")
	}
	defer eagerNormed.Free()

	runEager := func() {
		ctx.RMSNorm(input, weight, eagerNormed, rows, cols, 1e-5)
		ctx.Add(eagerNormed, bias, eager, rows*cols)
	}

	runEager()
	ctx.Synchronize()
	eagerHost := eager.ToHostF32()

	// The captured path: the same kernels, on the workspace buffers.
	manager := NewCUDAGraphManager(ctx)
	defer manager.Close()

	if err := manager.BeginCapture(1); err != nil {
		t.Fatalf("BeginCapture failed: %v", err)
	}
	ctx.RMSNorm(input, weight, workspace.Normed(), rows, cols, 1e-5)
	ctx.Add(workspace.Normed(), bias, workspace.AttnProj(), rows*cols)
	if err := manager.EndCapture(1); err != nil {
		t.Fatalf("EndCapture failed: %v", err)
	}

	if err := manager.Launch(1); err != nil {
		t.Fatalf("Launch failed: %v", err)
	}
	ctx.Synchronize()

	replayedHost := workspace.AttnProj().ToHostF32()
	if len(replayedHost) != len(eagerHost) {
		t.Fatalf("the replay produced %d values, the eager path %d", len(replayedHost), len(eagerHost))
	}
	for i := range eagerHost {
		if math.Abs(float64(replayedHost[i]-eagerHost[i])) > 1e-3 {
			t.Fatalf("the replay differs from the eager path at %d: %v and %v",
				i, replayedHost[i], eagerHost[i])
		}
	}

	// The replay uses the data that is in the buffers now, not the data
	// that was there when the graph was captured.
	newInput := make([]float32, rows*cols)
	for i := range newInput {
		newInput[i] = float32(i%5) * -0.25
	}
	if err := input.LoadFrom(newInput); err != nil {
		t.Fatalf("failed to reload the input: %v", err)
	}

	runEager()
	ctx.Synchronize()
	eagerHost = eager.ToHostF32()

	if err := manager.Launch(1); err != nil {
		t.Fatalf("the second Launch failed: %v", err)
	}
	ctx.Synchronize()
	replayedHost = workspace.AttnProj().ToHostF32()

	for i := range eagerHost {
		if math.Abs(float64(replayedHost[i]-eagerHost[i])) > 1e-3 {
			t.Fatalf("the second replay differs from the eager path at %d: %v and %v",
				i, replayedHost[i], eagerHost[i])
		}
	}
}
