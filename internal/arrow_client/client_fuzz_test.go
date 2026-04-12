package arrow_client

import (
	"context"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

// FuzzStreamEmbeddings ensures bounds and dimensionality panic safety 
// when streaming unaligned or malicious tensor payload counts.
func FuzzStreamEmbeddings(f *testing.F) {
	// Add some seed cases
	f.Add(uint(1), uint(128))
	f.Add(uint(10), uint(1024))
	
	f.Fuzz(func(t *testing.T, numTensors uint, cols uint) {
		if numTensors > 100 || cols > 4096 {
			return 
		}

		ctx := device.NewContext()
		defer ctx.Free()

		var tensors []*device.Tensor
		var ids []string

		for i := 0; i < int(numTensors); i++ {
			tensor := ctx.NewTensorFP32(1, int(cols))
			// Just blank tensor
			tensors = append(tensors, tensor)
			ids = append(ids, "test-id")
		}

		client, _ := NewFlightClient("localhost", 3000, "localhost", 3001)
		// Try streaming (expect flight client not connected or successful return, but not panic)
		_ = client.StreamEmbeddings(context.Background(), tensors, ids)
	})
}
