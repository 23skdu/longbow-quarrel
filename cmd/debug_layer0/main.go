package main

import (
	"fmt"
	"log"
	"math"
	"flag"

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/engine"
)

var modelPath = flag.String("model", "", "Path to GGUF model")

func main() {
	flag.Parse()
	if *modelPath == "" {
		log.Fatal("--model required")
	}

	// Load Engine (GPU mode implied by tags)
	fmt.Println("Loading Engine...")
	e, err := engine.NewEngine(*modelPath, false)
	if err != nil {
		log.Fatal(err)
	}
	defer e.Close()

	// Use Layer 0
	layerIdx := 0
	fmt.Printf("Debugging Layer %d...\n", layerIdx)

	// Create Input: Fixed pattern to be deterministic
	// Size: 1 x Dim
	rows := 1
	dim := e.Config.Dim
	
	input := make([]float32, dim)
	for i := range input {
		// Use a pattern that isn't just 0.1
		// Alternating signs, varying magnitude
		input[i] = float32(math.Sin(float64(i)*0.1)) * 0.1
	}
	
	// Create Tensor for Input (Must be F32 for Layer method)
	tInput := e.Ctx.NewTensorFP32(rows, dim) 
	tInput.LoadFromF32(input)

	// --- 1. GPU Execution ---
	fmt.Println("Running GPU Layer...")
	
	// Need Scratch
	scratch := e.Ctx.NewLayerScratch(1, dim, e.Config.HiddenDim, e.Config.Heads, e.Config.KVHeads, e.Config.HeadDim, 1, e.Config.VocabSize)
	defer scratch.Free()
	
	// Need KV Cache (Dummy)
	kvDim := e.Config.KVHeads * e.Config.HeadDim
	kCache := e.Ctx.NewTensor(1, kvDim) // 1 token capacity
	vCache := e.Ctx.NewTensor(1, kvDim)
	
	// Run Layer
	// func (t *Tensor) Layer(...)
	tInput.Layer(layerIdx, 
		e.Weights.AttnNorm[layerIdx],
		e.Weights.AttnQ[layerIdx],
		e.Weights.AttnK[layerIdx],
		e.Weights.AttnV[layerIdx],
		e.Weights.AttnO[layerIdx],
		e.Weights.FfnNorm[layerIdx],
		e.Weights.FfnGate[layerIdx],
		e.Weights.FfnUp[layerIdx],
		e.Weights.FfnDown[layerIdx],
		kCache, vCache,
		scratch,
		0, // CachePos
		e.Config.Heads, e.Config.KVHeads, e.Config.HeadDim,
		e.Config.RopeTheta, e.Config.Eps, e.Config.HiddenDim,
		1, // SeqLen
		e.GlobalScale,
	)
	
	e.Ctx.Synchronize()
	
	// DEBUG: Check KV Cache contents
	kCacheData := kCache.ToHost()
	vCacheData := vCache.ToHost()
	fmt.Printf("DEBUG: K Cache (first 10): %v\n", kCacheData[:10])
	fmt.Printf("DEBUG: V Cache (first 10): %v\n", vCacheData[:10])
	
	// gpuOutput := tInput.ToHost() // Read back modified input (Residual)
	
	// Read Intermediate Tensors from Scratch for finer Debugging
	gpuNormed := scratch.Normed.ToHost()
	gpuQ := scratch.QPart.ToHost()
	gpuK := scratch.KPart.ToHost()
	
	fmt.Printf("DEBUG: GPU Q (first 10): %v\n", gpuQ[:10])
	fmt.Printf("DEBUG: GPU K (first 10): %v\n", gpuK[:10])
	
	// --- 2. CPU Execution ---
	fmt.Println("Running CPU Reference...")
	
	// Helper to load weights safely from GGUF directly
	loadW := func(name string) []float32 {
		return engine.LoadWeightFromGGUF(e, name)
	}

	// 2a. RMSNorm
	normW := loadW(fmt.Sprintf("blk.%d.attn_norm.weight", layerIdx))
	cpuNormed := device.CPURMSNorm(input, normW, e.Config.Eps)
	
	// Compare Norm
	assertClose("RMSNorm", cpuNormed, gpuNormed, 1e-3)
	
	// 2b. Q/K/V Projections
	// wQ := loadW(fmt.Sprintf("blk.%d.attn_q.weight", layerIdx))

	// 2c. K/V Projections
	wK := loadW(fmt.Sprintf("blk.%d.attn_k.weight", layerIdx))
	cpuK := device.CPUMatMul(cpuNormed, wK, 1, e.Config.KVHeads * e.Config.HeadDim, e.Config.Dim)
	assertClose("K Projection", cpuK, gpuK, 5e-2)
	
	// wV
	// gpuV := scratch.VPart.ToHost() // Need to read GPU V
	// (Add GPU read above first)
	
	// Q, K, V are correct.
	// Now Check RoPE?
	// GPU RoPE is in-place on Q/K.
	// cpuQ is raw. We need to RoPE it.
	
	// Apply RoPE to CPU Q/K
	// RoPE(data, pos, heads, headDim, theta)
	// cpuQ_RoPE := device.CPURoPE(cpuQ, 0, e.Config.Heads, e.Config.HeadDim, e.Config.RopeTheta)
	// cpuK_RoPE := device.CPURoPE(cpuK, 0, e.Config.KVHeads, e.Config.HeadDim, e.Config.RopeTheta)
	
	// Check RoPE
	// GPU Q/K are modified in place? 
	// scratch.QPart is modified in place by Layer?
	// Yes, Layer calls QPart.RoPE().
	// So gpuQ in `gpuQ := scratch.QPart.ToHost()` (done after Sync) IS RoPE'd.
	// So `Q Projection` check ABOVE passed RoPE'd Q against Raw Q?
	// Wait. 
	// If GPU Q is RoPE'd. And CPU Q is Raw.
	// They shouldn't match!
	// Unless Pos=0 RoPE is Identity?
	// RoPE(pos=0).
	// theta = 10000^-0 = 1.
	// freq = 0.
	// cos(0)=1. sin(0)=0.
	// out = x*1 - y*0 = x.
	// YES. At Pos 0, RoPE is IDENTITY.
	// So my check passed because Pos=0.
	// This does NOT verify RoPE logic for Pos > 0.
	// But it implies MatMul is correct.
	
	// Let's check Attention Logic.
	// Attn = Softmax(Q.K) . V
	// We need V.

	
	// ... (Rest of layer)
	// For now, let's just check up to Q projection. 
	// If Q matches, then weights are correct.
}

func assertClose(name string, cpu, gpu []float32, tol float64) {
	if len(cpu) != len(gpu) {
		log.Fatalf("%s: Length mismatch %d vs %d", name, len(cpu), len(gpu))
	}
	
	// Stats
	var maxDiff float64
	var sumDiff float64
	for i := range cpu {
		diff := math.Abs(float64(cpu[i] - gpu[i]))
		if diff > maxDiff { maxDiff = diff }
		sumDiff += diff
	}
	avgDiff := sumDiff / float64(len(cpu))
	
	fmt.Printf("[%s] MaxDiff: %f AvgDiff: %f\n", name, maxDiff, avgDiff)
	
	if maxDiff > tol {
		fmt.Printf("FAIL: %s mismatch exceeding tolerance %f\n", name, tol)
		// Print sample
		for i := 0; i < 10; i++ {
			fmt.Printf("  [%d] CPU %f GPU %f\n", i, cpu[i], gpu[i])
		}
	} else {
		fmt.Printf("PASS: %s matches.\n", name)
	}
}
