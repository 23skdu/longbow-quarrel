//go:build cuda
package device

import (
	"fmt"
	"math"
	"testing"
)

func TestCUDA_StoreKV_TurboQuant(t *testing.T) {
	ctx, err := NewCUDAContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Free()

	heads := 4
	headDim := 256 // Match blockSize
	windowSize := 5
	qjlRows := 64

	// Setup rotation matrices in context
	rot, err := ctx.NewTensorFP32(headDim, headDim)
	if err != nil {
		t.Fatal(err)
	}
	rotData := make([]float32, headDim*headDim)
	for i := 0; i < headDim; i++ {
		rotData[i*headDim+i] = 1.0
	}
	rot.LoadFrom(rotData)

	qjl, err := ctx.NewTensorFP32(qjlRows, headDim)
	if err != nil {
		t.Fatal(err)
	}
	qjlData := make([]float32, qjlRows*headDim)
	qjl.LoadFrom(qjlData)

    globalCUDAContext = ctx
    c := &Context{cudaCtx: ctx}
    c.TQRotation = &Tensor{ctx: c, cudaPtr: rot, rows: headDim, cols: headDim, dataType: DataTypeF32}
    c.TQQJL = &Tensor{ctx: c, cudaPtr: qjl, rows: qjlRows, cols: headDim, dataType: DataTypeF32}

	kCache := c.NewTurboTensor(windowSize, heads*headDim, DataTypeTQ1_0, headDim, qjlRows)
	vCache := c.NewTurboTensor(windowSize, heads*headDim, DataTypeTQ1_0, headDim, qjlRows)

	kHost := make([]float32, heads*headDim)
	vHost := make([]float32, heads*headDim)
	for i := 0; i < heads*headDim; i++ {
		kHost[i] = float32(math.Cos(float64(i)))
		vHost[i] = float32(math.Sin(float64(i)))
	}

	kT, err := ctx.NewTensorFP32(1, heads*headDim)
	if err != nil {
		t.Fatal(err)
	}
	kT.LoadFrom(kHost)
    kt := &Tensor{ctx: c, cudaPtr: kT, rows: 1, cols: heads*headDim}

	vT, err := ctx.NewTensorFP32(1, heads*headDim)
	if err != nil {
		t.Fatal(err)
	}
	vT.LoadFrom(vHost)
    vt := &Tensor{ctx: c, cudaPtr: vT, rows: 1, cols: heads*headDim}

	pos := 1
	kt.StoreKV(vt, kCache, vCache, pos, heads, headDim, windowSize)
    ctx.Synchronize()

	// Test FetchKV
    // We need space for seqLen * heads * headDim * 2 (K and V) elements in F16
    totalElements := (pos + 1) * heads * headDim * 2
	kOutRT, err := ctx.NewTensorRaw(totalElements * 2) // 2 bytes per F16
	if err != nil {
		t.Fatal(err)
	}
    kOutRT.rows = (pos + 1) * 2
    kOutRT.cols = heads * headDim
    // Set dataType to F16 so FetchKV logic stays consistent
    kOut := &Tensor{ctx: c, cudaPtr: kOutRT, rows: kOutRT.rows, cols: kOutRT.cols, dataType: DataTypeF16}

	kOut.FetchKV(kCache, vCache, pos+1, heads, headDim)
    ctx.Synchronize()

    // KV are concatenated: [K_pos0, K_pos1, ..., V_pos0, V_pos1, ...]
    // We want K_pos1.
    allDecoded := kOutRT.ToHostF16AsF32()
    kDecoded := allDecoded[heads*headDim : 2*heads*headDim]

	// Since TQ is lossy, we only check that we got some non-zero values back
	// that roughly match the signs of the input.
	for i := 0; i < 4; i++ {
		if (kHost[i] > 0.1 && kDecoded[i] <= 0) || (kHost[i] < -0.1 && kDecoded[i] >= 0) {
			t.Errorf("FetchKV: numerical mismatch at k[%d]: got %f, want sign of %f", i, kDecoded[i], kHost[i])
		}
	}
    fmt.Printf("CUDA TQ StoreKV Test PASSED for index 0: got %f, input was %f\n", kDecoded[0], kHost[0])
}
