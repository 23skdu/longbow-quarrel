//go:build linux && amd64 && cuda && cgo

package device

import (
	"fmt"
	"sync"
	"unsafe"
)

// DecodeWorkspace holds the buffers a decode forward pass writes into.
//
// A captured CUDA graph bakes in the address of every buffer it touches,
// so the buffers a captured forward uses have to be allocated once, owned
// by the workspace, and reused for every replay. A forward that allocates
// its tensors per call would replay into whatever the pool handed out
// last time: stale pointers, silent corruption.
//
// The workspace is keyed by the shape it was built for. A decode pass
// with a different number of tokens needs a different workspace, and the
// graph is captured per shape as well.
type DecodeWorkspace struct {
	mu  sync.Mutex
	ctx *Context

	numTokens int
	dim       int
	hiddenDim int
	heads     int
	kvHeads   int
	headDim   int
	batchSize int
	maxBlocks int
	vocabSize int

	// The persistent buffers. Their addresses never change.
	tokenPos   *Tensor // I32 [1, numTokens]
	tokenToSeq *Tensor // I32 [1, numTokens]
	blockTable *Tensor // I32 [batchSize, maxBlocks]
	physPos    *Tensor // F32 [1, numTokens]
	hidden     *Tensor // F32 [numTokens, dim]
	normed     *Tensor // F32 [numTokens, dim]
	attnOut    *Tensor // F32 [numTokens, dim]
	normedFFN  *Tensor // F32 [numTokens, dim]
	attnProj   *Tensor // F32 [numTokens, dim]
	gate       *Tensor // F32 [numTokens, hiddenDim]
	up         *Tensor // F32 [numTokens, hiddenDim]
	down       *Tensor // F32 [numTokens, dim]
	logits     *Tensor // F32 [1, vocabSize]
}

// DecodeWorkspaceShape is the shape a workspace is built for.
type DecodeWorkspaceShape struct {
	NumTokens int
	Dim       int
	HiddenDim int
	Heads     int
	KVHeads   int
	HeadDim   int
	BatchSize int
	MaxBlocks int
	VocabSize int
}

// NewDecodeWorkspace allocates the persistent buffers of a decode pass.
//
// The buffers are unpooled on purpose: a pooled buffer can be handed to
// two goroutines, and its address can change when the pool recycles it.
func NewDecodeWorkspace(ctx *Context, shape DecodeWorkspaceShape) (*DecodeWorkspace, error) {
	if shape.NumTokens <= 0 || shape.Dim <= 0 || shape.VocabSize <= 0 {
		return nil, fmt.Errorf("cuda decode workspace: invalid shape %+v", shape)
	}
	if shape.BatchSize <= 0 {
		shape.BatchSize = 1
	}
	if shape.MaxBlocks <= 0 {
		shape.MaxBlocks = 1
	}

	w := &DecodeWorkspace{
		ctx:       ctx,
		numTokens: shape.NumTokens,
		dim:       shape.Dim,
		hiddenDim: shape.HiddenDim,
		heads:     shape.Heads,
		kvHeads:   shape.KVHeads,
		headDim:   shape.HeadDim,
		batchSize: shape.BatchSize,
		maxBlocks: shape.MaxBlocks,
		vocabSize: shape.VocabSize,
	}

	alloc := func(name string, rows, cols int, dtype DataType) (*Tensor, error) {
		t, err := ctx.NewTensor(rows, cols, dtype)
		if err != nil {
			return nil, fmt.Errorf("cuda decode workspace: %s: %w", name, err)
		}
		return t, nil
	}

	var err error
	if w.tokenPos, err = alloc("token positions", 1, shape.NumTokens, DataTypeI32); err != nil {
		return nil, err
	}
	if w.tokenToSeq, err = alloc("token to sequence", 1, shape.NumTokens, DataTypeI32); err != nil {
		w.Free()
		return nil, err
	}
	if w.blockTable, err = alloc("block table", shape.BatchSize, shape.MaxBlocks, DataTypeI32); err != nil {
		w.Free()
		return nil, err
	}
	if w.physPos, err = alloc("physical positions", 1, shape.NumTokens, DataTypeF32); err != nil {
		w.Free()
		return nil, err
	}
	if w.hidden, err = alloc("hidden", shape.NumTokens, shape.Dim, DataTypeF32); err != nil {
		w.Free()
		return nil, err
	}
	if w.normed, err = alloc("normed", shape.NumTokens, shape.Dim, DataTypeF32); err != nil {
		w.Free()
		return nil, err
	}
	if w.attnOut, err = alloc("attention output", shape.NumTokens, shape.Dim, DataTypeF32); err != nil {
		w.Free()
		return nil, err
	}
	if w.normedFFN, err = alloc("normed feed forward", shape.NumTokens, shape.Dim, DataTypeF32); err != nil {
		w.Free()
		return nil, err
	}
	if w.attnProj, err = alloc("attention projection", shape.NumTokens, shape.Dim, DataTypeF32); err != nil {
		w.Free()
		return nil, err
	}
	if shape.HiddenDim > 0 {
		if w.gate, err = alloc("gate", shape.NumTokens, shape.HiddenDim, DataTypeF32); err != nil {
			w.Free()
			return nil, err
		}
		if w.up, err = alloc("up", shape.NumTokens, shape.HiddenDim, DataTypeF32); err != nil {
			w.Free()
			return nil, err
		}
	}
	if w.down, err = alloc("down", shape.NumTokens, shape.Dim, DataTypeF32); err != nil {
		w.Free()
		return nil, err
	}
	if w.logits, err = alloc("logits", 1, shape.VocabSize, DataTypeF32); err != nil {
		w.Free()
		return nil, err
	}

	return w, nil
}

// Shape returns the shape the workspace was built for.
func (w *DecodeWorkspace) Shape() DecodeWorkspaceShape {
	if w == nil {
		return DecodeWorkspaceShape{}
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return DecodeWorkspaceShape{
		NumTokens: w.numTokens,
		Dim:       w.dim,
		HiddenDim: w.hiddenDim,
		Heads:     w.heads,
		KVHeads:   w.kvHeads,
		HeadDim:   w.headDim,
		BatchSize: w.batchSize,
		MaxBlocks: w.maxBlocks,
		VocabSize: w.vocabSize,
	}
}

// Compatible reports whether the workspace can be used for a shape.
func (w *DecodeWorkspace) Compatible(shape DecodeWorkspaceShape) bool {
	if w == nil {
		return false
	}
	w.mu.Lock()
	defer w.mu.Unlock()

	if shape.BatchSize <= 0 {
		shape.BatchSize = 1
	}
	if shape.MaxBlocks <= 0 {
		shape.MaxBlocks = 1
	}

	return w.numTokens == shape.NumTokens &&
		w.dim == shape.Dim &&
		w.batchSize == shape.BatchSize &&
		w.maxBlocks == shape.MaxBlocks &&
		w.vocabSize == shape.VocabSize
}

// Addresses returns the address of every buffer of the workspace.
//
// A captured graph replays into these addresses, so they have to be the
// same on every call.
func (w *DecodeWorkspace) Addresses() map[string]uintptr {
	if w == nil {
		return nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()

	out := map[string]uintptr{}
	for name, t := range map[string]*Tensor{
		"tokenPos":   w.tokenPos,
		"tokenToSeq": w.tokenToSeq,
		"blockTable": w.blockTable,
		"physPos":    w.physPos,
		"hidden":     w.hidden,
		"normed":     w.normed,
		"attnOut":    w.attnOut,
		"normedFFN":  w.normedFFN,
		"attnProj":   w.attnProj,
		"gate":       w.gate,
		"up":         w.up,
		"down":       w.down,
		"logits":     w.logits,
	} {
		if t == nil {
			continue
		}
		out[name] = uintptr(unsafe.Pointer(t.devPtr))
	}
	return out
}

// TokenPos returns the persistent token position tensor.
func (w *DecodeWorkspace) TokenPos() *Tensor { return w.tokenPos }

// TokenToSeq returns the persistent token-to-sequence tensor.
func (w *DecodeWorkspace) TokenToSeq() *Tensor { return w.tokenToSeq }

// BlockTable returns the persistent block table tensor.
func (w *DecodeWorkspace) BlockTable() *Tensor { return w.blockTable }

// PhysPos returns the persistent physical position tensor.
func (w *DecodeWorkspace) PhysPos() *Tensor { return w.physPos }

// Hidden returns the persistent hidden state tensor.
func (w *DecodeWorkspace) Hidden() *Tensor { return w.hidden }

// Normed returns the persistent normalised activation tensor.
func (w *DecodeWorkspace) Normed() *Tensor { return w.normed }

// AttnOut returns the persistent attention output tensor.
func (w *DecodeWorkspace) AttnOut() *Tensor { return w.attnOut }

// NormedFFN returns the persistent feed-forward input tensor.
func (w *DecodeWorkspace) NormedFFN() *Tensor { return w.normedFFN }

// AttnProj returns the persistent attention projection tensor.
func (w *DecodeWorkspace) AttnProj() *Tensor { return w.attnProj }

// Gate returns the persistent feed-forward gate tensor.
func (w *DecodeWorkspace) Gate() *Tensor { return w.gate }

// Up returns the persistent feed-forward up tensor.
func (w *DecodeWorkspace) Up() *Tensor { return w.up }

// Down returns the persistent feed-forward down tensor.
func (w *DecodeWorkspace) Down() *Tensor { return w.down }

// Logits returns the persistent logits tensor.
func (w *DecodeWorkspace) Logits() *Tensor { return w.logits }

// Free releases every buffer of the workspace.
func (w *DecodeWorkspace) Free() {
	if w == nil {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()

	for _, t := range []*Tensor{
		w.tokenPos, w.tokenToSeq, w.blockTable, w.physPos,
		w.hidden, w.normed, w.attnOut, w.normedFFN, w.attnProj,
		w.gate, w.up, w.down, w.logits,
	} {
		if t != nil {
			t.Free()
		}
	}
	w.tokenPos = nil
	w.tokenToSeq = nil
	w.blockTable = nil
	w.physPos = nil
	w.hidden = nil
	w.normed = nil
	w.attnOut = nil
	w.normedFFN = nil
	w.attnProj = nil
	w.gate = nil
	w.up = nil
	w.down = nil
	w.logits = nil
}
