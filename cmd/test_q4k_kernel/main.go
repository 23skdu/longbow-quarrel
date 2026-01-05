//go:build darwin && metal

package main

import (
	"fmt"

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/engine"
)

func main() {
	modelPath := "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
	e, err := engine.NewEngine(modelPath, false)
	if err != nil {
		panic(err)
	}

	// Get Q4K Weight (Attn Q Layer 0)
	w := e.Weights.AttnQ[0]
	if w == nil {
		panic("No AttnQ[0]")
	}
	fmt.Printf("Weight: %dx%d Type=%v\n", w.Rows(), w.Cols(), w.DataType())

	// Input (F16)
	ctx := e.Ctx
	rows := 1        // One token
	cols := w.Cols() // 4096
	input := ctx.NewTensorWithType(rows, cols, device.DataTypeF16)

	f32In := make([]float32, cols)
	for i := range f32In {
		f32In[i] = 0.01
	}
	input.LoadFrom(f32In)

	// Output (F16) - Layers use F16->F16
	output := ctx.NewTensorWithType(rows, w.Rows(), device.DataTypeF16)

	// Run Kernel: LinearInto
	// input.LinearInto(w, output, 1.0)
	// Method: func (t *Tensor) LinearInto(weight *Tensor, dst *Tensor, scale float32)
	input.LinearInto(w, output, 1.0)

	ctx.Synchronize()

	res := output.ToHost() // Returns []float32 (converting from F16)

	fmt.Printf("Result[0..10]: %v\n", res[:10])

	max := float32(0)
	for _, v := range res {
		if v > max {
			max = v
		}
		if -v > max {
			max = -v
		}
	}
	fmt.Printf("Max Logic: %f\n", max)
}
