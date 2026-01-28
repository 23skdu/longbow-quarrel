//go:build darwin && metal

package engine

import (
	"github.com/23skdu/longbow-quarrel/internal/device"
)

// MambaWeights holds the weights for a single Mamba/SSM layer
// Based on Nemotron-3-Nano tensor names:
// blk.N.ssm_a
// blk.N.ssm_conv1d.weight/bias
// blk.N.ssm_d
// blk.N.ssm_dt.bias (and potentially weight if present, though log only showed bias)
// blk.N.ssm_norm.weight
// blk.N.ssm_out.weight
type MambaWeights struct {
	// A: State transition parameter (S4/Mamba A matrix, usually diagonal in Mamba)
	// Shape: [d_inner, d_state]
	A *device.Tensor

	// D: Skip connection (D parameter)
	// Shape: [d_inner]
	D *device.Tensor

	// Conv1d: 1D Convolution weights and bias
	// Shape: [d_inner, 1, kernel_size] (Depthwise coonvolution)
	Conv1dWeight *device.Tensor
	Conv1dBias   *device.Tensor

	// DT: Delta (Step size) projection
	// Usually projects from [d_inner] -> [d_inner]
	// Log only showed bias, but standard Mamba has weight too. We'll support both.
	DTWeight *device.Tensor
	DTBias   *device.Tensor

	// Norm: Normalization before/after SSM?
	// Nemotron has `ssm_norm.weight`
	NormWeight *device.Tensor
	NormBias   *device.Tensor // Optional

	// Out: Output projection
	// Projects [d_inner] -> [hidden_dim]
	OutWeight *device.Tensor

	// In: Input projection (x -> [z, x_ssm])
	// Nemotron logs didn't show `ssm_in`. It might be missing or named differently.
	// We will add it to the struct for completeness/future proofing.
	InWeight *device.Tensor
}

func (w *MambaWeights) Free() {
	if w == nil {
		return
	}
	if w.A != nil {
		w.A.Free()
	}
	if w.D != nil {
		w.D.Free()
	}
	if w.Conv1dWeight != nil {
		w.Conv1dWeight.Free()
	}
	if w.Conv1dBias != nil {
		w.Conv1dBias.Free()
	}
	if w.DTWeight != nil {
		w.DTWeight.Free()
	}
	if w.DTBias != nil {
		w.DTBias.Free()
	}
	if w.NormWeight != nil {
		w.NormWeight.Free()
	}
	if w.NormBias != nil {
		w.NormBias.Free()
	}
	if w.OutWeight != nil {
		w.OutWeight.Free()
	}
	if w.InWeight != nil {
		w.InWeight.Free()
	}
}

// IsMambaLayer checks if a layer index corresponds to a Mamba layer
// for a hybrid model. This depends on the specific architecture (interleaving pattern).
func (e *Engine) IsMambaLayer(layerIdx int) bool {
	// TODO: Implement logic based on model config.
	// For Nemotron-3-Nano, it seems to be block 0, 2, 4... (Even layers?)
	// or specific pattern. We need to detect this during loading.
	// For now, checks if we have Mamba weights loaded for this layer.
	if layerIdx < len(e.Weights.Mamba) {
		return e.Weights.Mamba[layerIdx] != nil // Non-nil means Mamba weights exist
	}
	return false
}

// MambaState holds the recurrent state for a single Mamba layer
type MambaState struct {
	// ConvState: Ring buffer for 1D convolution
	// Shape: [d_conv, kernel_size]
	// Typically [d_inner, 4]
	ConvState *device.Tensor
	ConvStep  int // Current position in ring buffer? Or we shift?
	// Note: Metal kernel often uses ring buffer index derived from global pos.

	// SSMState: Hidden state for SSM scan
	// Shape: [d_ssm, d_state]
	// Typically [d_inner, 16] or [d_inner, 64]
	SSMState *device.Tensor
}

func (s *MambaState) Free() {
	if s == nil {
		return
	}
	if s.ConvState != nil {
		s.ConvState.Free()
	}
	if s.SSMState != nil {
		s.SSMState.Free()
	}
}
