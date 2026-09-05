//go:build cuda
package device

import (
	"testing"
)

func TestCUDA_StoreKV_TurboQuant(t *testing.T) {
	t.Skip("Requires full KV cache infrastructure - FetchKV not yet implemented for CUDA")
}

