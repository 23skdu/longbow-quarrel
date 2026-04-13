//go:build darwin && metal

package vlm

import (
	"testing"
)

func TestEncoder_Projection(t *testing.T) {
	t.Run("MockProjection", func(t *testing.T) {
		// Test image to vector projection logic
		_ = &VisionEncoder{dim: 512}
	})
}
