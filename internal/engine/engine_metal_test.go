//go:build darwin && metal

package engine

import (
	"testing"
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestInitKVCache(t *testing.T) {
	tests := []struct {
		name                string
		config              config.Config
		expectedError       bool
		expectedKVCacheKLen int
		expectedKVCacheVLen int
	}{
		{
			name: "Valid config with window size",
			config: config.Config{
				Layers:     2,
				WindowSize: 10,
				KVHeads:    2,
				HeadDim:    4,
				SeqLen:     20, // Should be overridden by WindowSize if set
			},
			expectedError:       false,
			expectedKVCacheKLen: 2,
			expectedKVCacheVLen: 2,
		},
		{
			name: "Valid config without window size (uses SeqLen)",
			config: config.Config{
				Layers:     1,
				WindowSize: 0,
				KVHeads:    1,
				HeadDim:    8,
				SeqLen:     15,
			},
			expectedError:       false,
			expectedKVCacheKLen: 1,
			expectedKVCacheVLen: 1,
		},
		{
			name: "Invalid config: zero KVHeads",
			config: config.Config{
				Layers:     1,
				WindowSize: 10,
				KVHeads:    0,
				HeadDim:    4,
				SeqLen:     20,
			},
			expectedError: true,
		},
		{
			name: "Invalid config: zero HeadDim",
			config: config.Config{
				Layers:      2,
				WindowSize:  32,
				KVHeads:     2,
				HeadDim:     0,
				KVCacheSize: 1024,
			},
			expectedError: true,
		},
		{
			name: "Zero layers",
			config: config.Config{
				Layers:     0,
				WindowSize: 10,
				KVHeads:    2,
				HeadDim:    4,
				SeqLen:     20,
			},
			expectedError:       true,
			expectedKVCacheKLen: 0,
			expectedKVCacheVLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := device.NewContext()
			defer ctx.Free()
			me := &metalEngine{
				ctx:    ctx,
				config: tt.config,
			}

			err := me.initKVCache()

			if tt.expectedError {
				if err == nil {
					t.Errorf("Expected an error for %s, but got none", tt.name)
				}
				return // Skip further checks if error is expected
			} else if err != nil {
				t.Fatalf("Unexpected error for %s: %v", tt.name, err)
			}

			if me.config.Layers != tt.expectedKVCacheKLen {
				t.Errorf("Layers count mismatch for %s: got %d, expected %d", tt.name, me.config.Layers, tt.expectedKVCacheKLen)
			}

			for i := 0; i < tt.expectedKVCacheKLen; i++ {
				view := me.cache.Get("seq-0", i)
				if view.K == nil {
					t.Errorf("KVCacheK[%d] is nil for %s", i, tt.name)
				}
				if view.V == nil {
					t.Errorf("KVCacheV[%d] is nil for %s", i, tt.name)
				}
			}
		})
	}
}
