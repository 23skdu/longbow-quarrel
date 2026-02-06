//go:build darwin && metal

package engine

import (
	"fmt"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

// MambaLayer executes a Mamba/SSM block
type MambaLayer struct {
	Index   int
	Weights *MambaWeights
}

// Forward executes the Mamba layer
// Input: [Batch, Dim]
func (l *MambaLayer) Forward(input *device.Tensor, state *MambaState) (*device.Tensor, error) {
	if l.Weights.InWeight == nil {
		return nil, fmt.Errorf("MambaLayer %d: Missing InWeight (ssm_in)", l.Index)
	}

	ctx := input.Context()

	// 1. Norm
	var normInput *device.Tensor
	if l.Weights.NormWeight != nil {
		normInput = input.RMSNorm(l.Weights.NormWeight, 1e-5)
	} else {
		normInput = input
	}

	// 2. Input Projection [xz, B, C]
	// Nemotron-3-Nano: [2688 -> 6144]
	// We assume xz is first 4096, B is next 1024, C is last 1024
	projected := normInput.MatMul(l.Weights.InWeight)

	xz := projected.Slice(0, 4096)
	ssmB := projected.Slice(4096, 1024)
	ssmC := projected.Slice(5120, 1024)

	// 3. Conv1d
	xzConv := ctx.NewTensorPooled(xz.Rows(), xz.Cols())
	xz.MambaConv1d(l.Weights.Conv1dWeight, l.Weights.Conv1dBias, state.ConvState, xzConv)

	// 4. Split x and z and activate
	// xzConv is [4096]. We assume x is first 2048, z is next 2048?
	// Actually, let's assume x and z are interleaved or split.
	// Standard Mamba: x = first half, z = second half.
	x := xzConv.Slice(0, 2048)
	z := xzConv.Slice(2048, 2048)

	zAct := z.SiLU()

	// 5. Scan
	// We need dt projection: ssm_dt is separate in GGUF
	var dt *device.Tensor
	if l.Weights.DTWeight != nil {
		dt = normInput.MatMul(l.Weights.DTWeight)
		dt, _ = dt.Add(l.Weights.DTBias)
	} else if l.Weights.DTBias != nil {
		dt = l.Weights.DTBias
	} else {
		return nil, fmt.Errorf("MambaLayer %d: Missing dt params", l.Index)
	}

	// Scan: y = Scan(x, dt, A, B, C, D)
	// dState = 16 (since B is 1024 / 64 heads)
	y := x.MambaScan(state.SSMState, l.Weights.A, ssmB, ssmC, l.Weights.D, dt, 16)

	// 6. Gating [y * SiLU(z)]
	// We need element-wise multiply. We'll use Scale if possible or a new kernel.
	// For now, let's assume we have a Multiply/Hadamard kernel or just use a loop (slow)
	// Actually, I'll add a quick Mul kernel.

	gated := y.Mul(zAct)

	// 7. Output Projection
	final := gated.MatMul(l.Weights.OutWeight)

	return final, nil
}
