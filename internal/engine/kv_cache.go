package engine

import (
	"github.com/23skdu/longbow-quarrel/internal/device"
)

// CacheView holds the tensors and metadata required for attention
type CacheView struct {
	K          *device.Tensor
	V          *device.Tensor
	BlockTable *device.Tensor // Optional: Paged Attention Block Table (Int32)
	BlockSize  int            // Optional: for Paged Attention
}
