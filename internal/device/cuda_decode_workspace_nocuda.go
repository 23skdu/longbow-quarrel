//go:build !linux || !amd64 || !cuda || !cgo

package device

import "fmt"

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

// DecodeWorkspace holds the buffers a decode forward pass writes into.
//
// The CUDA graph capture needs the buffers of a captured forward to be
// allocated once and reused, so that the replay writes to the same
// addresses. This build has no CUDA, so there is nothing to hold.
type DecodeWorkspace struct{}

// NewDecodeWorkspace is not available in this build.
func NewDecodeWorkspace(ctx *Context, shape DecodeWorkspaceShape) (*DecodeWorkspace, error) {
	return nil, fmt.Errorf("cuda decode workspace: CUDA is not available in this build")
}

// Shape returns the shape the workspace was built for.
func (w *DecodeWorkspace) Shape() DecodeWorkspaceShape { return DecodeWorkspaceShape{} }

// Compatible reports whether the workspace can be used for a shape.
func (w *DecodeWorkspace) Compatible(shape DecodeWorkspaceShape) bool { return false }

// Addresses returns the address of every buffer of the workspace.
func (w *DecodeWorkspace) Addresses() map[string]uintptr { return nil }

// TokenPos returns the persistent token position tensor.
func (w *DecodeWorkspace) TokenPos() *Tensor { return nil }

// TokenToSeq returns the persistent token-to-sequence tensor.
func (w *DecodeWorkspace) TokenToSeq() *Tensor { return nil }

// BlockTable returns the persistent block table tensor.
func (w *DecodeWorkspace) BlockTable() *Tensor { return nil }

// PhysPos returns the persistent physical position tensor.
func (w *DecodeWorkspace) PhysPos() *Tensor { return nil }

// Hidden returns the persistent hidden state tensor.
func (w *DecodeWorkspace) Hidden() *Tensor { return nil }

// Normed returns the persistent normalised activation tensor.
func (w *DecodeWorkspace) Normed() *Tensor { return nil }

// AttnOut returns the persistent attention output tensor.
func (w *DecodeWorkspace) AttnOut() *Tensor { return nil }

// NormedFFN returns the persistent feed-forward input tensor.
func (w *DecodeWorkspace) NormedFFN() *Tensor { return nil }

// AttnProj returns the persistent attention projection tensor.
func (w *DecodeWorkspace) AttnProj() *Tensor { return nil }

// Gate returns the persistent feed-forward gate tensor.
func (w *DecodeWorkspace) Gate() *Tensor { return nil }

// Up returns the persistent feed-forward up tensor.
func (w *DecodeWorkspace) Up() *Tensor { return nil }

// Down returns the persistent feed-forward down tensor.
func (w *DecodeWorkspace) Down() *Tensor { return nil }

// Logits returns the persistent logits tensor.
func (w *DecodeWorkspace) Logits() *Tensor { return nil }

// Free releases every buffer of the workspace.
func (w *DecodeWorkspace) Free() {}
