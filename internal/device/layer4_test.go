package device

import (
	"math/rand"
	"testing"
	"time"
)

func TestLayer4_Q4KMatMul(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Dimensions
	M := 1
	K := 256 // 1 Block
	N := 1
	
	// 1. Generate Q4K Data (Bytes)
	// We need valid Q4K blocks.
	// 1 block = 144 bytes.
	// 0-2: d
	// 2-4: dmin
	// 4-16: scales
	// 16-144: qs
	
	q4kData := make([]byte, (N*K/256)*144)
	
	// Scale D and Dmin
	d := Float32ToFloat16(0.005)
	dmin := Float32ToFloat16(0.0) // Simplest case
	
	for i := 0; i < len(q4kData)/144; i++ {
		off := i * 144
		q4kData[off] = byte(d & 0xFF)
		q4kData[off+1] = byte(d >> 8)
		q4kData[off+2] = byte(dmin & 0xFF)
		q4kData[off+3] = byte(dmin >> 8)
		
		// Random scales and qs
		for j := 4; j < 144; j++ {
			q4kData[off+j] = byte(rand.Intn(256))
		}
	}
	
	// 2. Generate Input (F32)
	input := make([]float32, M*K)
	for i := range input {
		input[i] = (rand.Float32() - 0.5)
	}
	
	// 3. CPU Reference
	cpuOut := CPUQ4KMatMul(input, q4kData, M, N, K)
	
	// 4. GPU Implementation
	// Load Input (F16)
	tInput := ctx.NewTensor(M, K)
	tInput.LoadFrom(input) // Converts F32->F16 in LoadFrom usually if tensor is F16?
	// Wait, LoadFrom takes float32 and converts based on tensor dtype.
	// NewTensor defaults to F16.
	
	// Load Weights (Q4K)
	// We need a tensor with Q4K type.
	// Manually create? Or use NewTensor + SetDataType?
	// NewTensor allocates F16 buffer size.
	// Q4K is smaller.
	// Let's use NewTensorFromBytes if exists, or manual
	
	// Helper to load Q4K
	tWeight := ctx.NewTensor(N, K)
	// We must hack the buffer size or reuse NewTensor functionality if it supports Q4K?
	// internal/device/metal.go: NewTensor allocates based on rows*cols*sizeof(half).
	// Q4K needs specialized allocation?
	// Actually, `Linear` expects the weight tensor to be properly set up.
	// Let's inspect `NewTensor` logic or manually alloc.
	// For now, we can overwrite the buffer contents if size is sufficient (F16 is 2 bytes/elem, Q4K is ~0.56 bytes/elem, so F16 buffer is big enough).
	// But `ToGPU` or `LoadFrom` might struggle.
	// `LoadFrom` expects float32 slice.
	// We need raw byte upload.
	
	// Copy bytes to buffer
	// Copy bytes to buffer
	// Implement simple copy
	// tWeight.CopyBytes(q4kData) // Not existing
	
	// Workaround: We implemented `NewTensor` but not generic byte loader.
	// Let's use `tWeight.LoadFrom` with dummy data just to init, then overwrite buffer?
	// No, that overwrites with F16 logic.
	
	// We need to implement a helper to load Q4K data or expose buffer write.
	// Assuming check `TestLayer4_Q4KMatMul` can access internals or we add helper.
	
	// We need to set dataType field using reflection or if exposed.
	// Actually, `Tensor` struct in `device` package has `dataType` field.
	// Check if it is exported. It is `dataType` in `metal.go`, lowercase?
	// view_file metal.go showed `dataType DataType`.
	// If it is unexported, we can't set it from test unless test is in `device` package.
	// We are in `package device`, so we can access unexported fields!
	tWeight.dataType = DataTypeQ4K // Set type
	
	// Copy bytes to buffer
	// But `NewTensor` fixes size.
	// Since F16 > Q4K, we can just use the memory.
	
	// Hack: Use `LoadFrom` with `nil`? No.
	// Helper in test:
	CopyToBuffer(tWeight, q4kData)
	
	// Run Linear
	tOut := tInput.Linear(tWeight)
	tOut.ctx.Synchronize()
	gpuOut := tOut.ToHost()
	
	// 5. Compare
	// Q4K dot product on GPU is F16 accumulation + complex decoding.
	// Tolerance might be higher.
	assertClose(t, "Q4K MatMul", cpuOut, gpuOut, 1e-1) // 1e-1 is loose but detects "garbage"
}

// Helper
func CopyToBuffer(t *Tensor, data []byte) {
	// Use LoadFromBytes which properly calls Metal_CopyToDevice
	t.LoadFromBytes(data)
}

func TestLayer4_Q4KMatMul_FP32(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Dimensions
	M := 1
	K := 256
	// Guardrail: Check dimensions
	if M*K > 1_000_000 && testing.Short() {
		t.Skip("Skipping large tensor test in short mode")
	}
	if int64(M)*int64(K) > 100_000_000 { // 100M elements ~ 200MB/400MB
		t.Fatal("Test dimension guardrail exceeded safe limit")
	}
	N := 1
	
	// 1. Generate Q4K Data
	q4kData := make([]byte, (N*K/256)*144)
	d := Float32ToFloat16(0.005)
	dmin := Float32ToFloat16(0.0)
	
	for i := 0; i < len(q4kData)/144; i++ {
		off := i * 144
		q4kData[off] = byte(d & 0xFF)
		q4kData[off+1] = byte(d >> 8)
		q4kData[off+2] = byte(dmin & 0xFF)
		q4kData[off+3] = byte(dmin >> 8)
		
		// Random scales and qs
		for j := 4; j < 144; j++ {
			q4kData[off+j] = byte(rand.Intn(256))
		}
	}
	
	// 2. Generate Input (F32)
	input := make([]float32, M*K)
	for i := range input {
		input[i] = (rand.Float32() - 0.5)
	}
	
	// 3. CPU Reference (Same function, it handles F32 A, Q4K B)
	cpuOut := CPUQ4KMatMul(input, q4kData, M, N, K)
	
	// 4. GPU Implementation (FP32 Path)
	// Input Tensor: F32
	tInput := ctx.NewTensorFP32(M, K) // Implemented in previous session? Check metal.go if NewTensorFP32 exists.
	// Checked: yes, `func (c *Context) NewTensorFP32`.
	tInput.LoadFrom(input) // LoadFrom handles conversion? LoadFrom accepts []float32.
	// If tensor is FP32, it copies directly.
	
	tWeight := ctx.NewTensor(N, K)
	CopyToBuffer(tWeight, q4kData)
	tWeight.dataType = DataTypeQ4K
	ctx.Synchronize() // Ensure weight data is copied before kernel runs
	
	// Output: F32
	tOut := ctx.NewTensorFP32(M, N)
	
	// Explicitly call LinearIntoFP32 (if exposed) or LinearFP32
	// metal.go: `func (t *Tensor) LinearFP32(weight *Tensor, out *Tensor)`?
	// grep showed `LinearIntoFP32` on line 717.
	// Let's use that.
	
	tInput.LinearIntoFP32(tWeight, tOut)
	
	// Use safer synchronization with timeout
	if err := ctx.WaitWithTimeout(5 * time.Second); err != nil {
		t.Fatal(err)
	}
	gpuOut := tOut.ToHost()
	
	// Manual Cleanup
	tInput.Free()
	tWeight.Free()
	tOut.Free()
	
	// 5. Compare
	assertClose(t, "Q4K MatMul FP32", cpuOut, gpuOut, 1.0)
}
