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

	// Get Output Weight (Q6K)
	w := e.Weights.Output // 32000 x 4096
	// Or OutputNorm? No, use Output Weight.
	if w == nil {
		panic("No output weight")
	}
	fmt.Printf("Weight: %dx%d Type=%v\n", w.Rows(), w.Cols(), w.DataType())

	// Input (F16)
	ctx := e.Ctx
	rows := 1
	cols := w.Cols()
	input := ctx.NewTensorWithType(rows, cols, device.DataTypeF16)

	// Set Input to 1.0 everywhere? Or random?
	// Use 1.0 to check scale.
	f32In := make([]float32, cols)
	for i := range f32In {
		f32In[i] = 0.01
	} // 0.01
	input.LoadFrom(f32In)

	// Output (F32)
	output := ctx.NewTensorFP32(rows, w.Rows())

	// Run Kernel
	// Using "LinearToFP32_Into" logic manually to invoke specific kernel path?
	// input.LinearToFP32_Into(w, output) handles F16 input?
	// Yes: func (t *Tensor) LinearToFP32_Into(w *Tensor, dest *Tensor)
	// It dispatches Metal_LinearQ6K_F16_F32 if w is Q6K.
	input.LinearToFP32_Into(w, output)

	ctx.Synchronize()

	res := output.ToHost()

	// Check results
	// For row 0 of W: dot(w[0], input)
	// W is on GPU (Q6K). We can't access it easily on host to verify.
	// But we can check if result is NaN, 0, or plausible.
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
