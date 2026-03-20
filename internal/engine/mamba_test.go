//go:build darwin && metal

package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestMambaWeightsStructure(t *testing.T) {
	mw := &MambaWeights{
		A:            &device.Tensor{},
		D:            &device.Tensor{},
		Conv1dWeight: &device.Tensor{},
	}

	if mw.A == nil {
		t.Error("MambaWeights.A should not be nil")
	}
}

func TestIsMambaLayer(t *testing.T) {
	tests := []struct {
		name     string
		layers   int
		pattern  string
		isHybrid bool
		mambaIdx []int
		layer    int
		want     bool
	}{
		{
			name:    "pure mamba - all layers",
			layers:  10,
			pattern: "all",
			layer:   5,
			want:    true,
		},
		{
			name:    "pure transformer - no mamba",
			layers:  10,
			pattern: "none",
			layer:   5,
			want:    false,
		},
		{
			name:    "hybrid even pattern - mamba layer",
			layers:  10,
			pattern: "even",
			layer:   0,
			want:    true,
		},
		{
			name:    "hybrid even pattern - transformer layer",
			layers:  10,
			pattern: "even",
			layer:   1,
			want:    false,
		},
		{
			name:    "hybrid odd pattern - mamba layer",
			layers:  10,
			pattern: "odd",
			layer:   1,
			want:    true,
		},
		{
			name:    "hybrid odd pattern - transformer layer",
			layers:  10,
			pattern: "odd",
			layer:   0,
			want:    false,
		},
		{
			name:     "weight-based detection - mamba weight exists",
			layers:   10,
			mambaIdx: []int{2, 5, 8},
			layer:    2,
			want:     true,
		},
		{
			name:     "weight-based detection - no mamba weight",
			layers:   10,
			mambaIdx: []int{2, 5, 8},
			layer:    3,
			want:     false,
		},
		{
			name:    "out of bounds",
			layers:  10,
			pattern: "all",
			layer:   99,
			want:    false,
		},
		{
			name:    "negative index",
			layers:  10,
			pattern: "all",
			layer:   -1,
			want:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &Engine{
				Config: config.Config{
					Layers:            tt.layers,
					MambaLayerPattern: tt.pattern,
					IsHybrid:          tt.isHybrid,
				},
				Weights: &LlamaWeights{
					Mamba: make([]*MambaWeights, tt.layers),
				},
			}

			// Set up Mamba weights for weight-based tests
			for _, idx := range tt.mambaIdx {
				if idx < tt.layers {
					e.Weights.Mamba[idx] = &MambaWeights{}
				}
			}

			got := e.IsMambaLayer(tt.layer)
			if got != tt.want {
				t.Errorf("IsMambaLayer(%d) = %v, want %v", tt.layer, got, tt.want)
			}
		})
	}
}

func TestCountMambaLayers(t *testing.T) {
	tests := []struct {
		name     string
		layers   int
		pattern  string
		isHybrid bool
		mambaIdx []int
		want     int
	}{
		{
			name:    "pure mamba",
			layers:  10,
			pattern: "all",
			want:    10,
		},
		{
			name:    "pure transformer",
			layers:  10,
			pattern: "none",
			want:    0,
		},
		{
			name:    "hybrid even pattern",
			layers:  10,
			pattern: "even",
			want:    5,
		},
		{
			name:    "hybrid odd pattern",
			layers:  10,
			pattern: "odd",
			want:    5,
		},
		{
			name:     "weight-based - specific layers",
			layers:   10,
			mambaIdx: []int{0, 2, 4, 6, 8},
			want:     5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &Engine{
				Config: config.Config{
					Layers:            tt.layers,
					MambaLayerPattern: tt.pattern,
					IsHybrid:          tt.isHybrid,
				},
				Weights: &LlamaWeights{
					Mamba: make([]*MambaWeights, tt.layers),
				},
			}

			for _, idx := range tt.mambaIdx {
				if idx < tt.layers {
					e.Weights.Mamba[idx] = &MambaWeights{}
				}
			}

			got := e.CountMambaLayers()
			if got != tt.want {
				t.Errorf("CountMambaLayers() = %d, want %d", got, tt.want)
			}
		})
	}
}
