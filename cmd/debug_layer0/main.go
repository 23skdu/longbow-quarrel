//go:build darwin && metal

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
	
	// Use adequate capacity for debugging
	debugSeqLen := 128 
	scratch := e.Ctx.NewLayerScratch(1, dim, e.Config.HiddenDim, e.Config.Heads, e.Config.KVHeads, e.Config.HeadDim, debugSeqLen, e.Config.VocabSize)
	defer scratch.Free()
	
	// Need KV Cache (Dummy)
	kvDim := e.Config.KVHeads * e.Config.HeadDim
	// Must be large enough for debugSeqLen positions!
	kCache := e.Ctx.NewTensor(debugSeqLen, kvDim) 
	vCache := e.Ctx.NewTensor(debugSeqLen, kvDim)
	
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
		debugSeqLen, // SeqLen (Stride)
		e.Config.WindowSize,
		e.GlobalScale,
		true, // DebugActivations
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
	fmt.Printf("RMSNorm Sample: CPU %v GPU %v\n", cpuNormed[:4], gpuNormed[:4])
	
	wQ := loadW(fmt.Sprintf("blk.%d.attn_q.weight", layerIdx))
	cpuQ := device.CPUMatMul(cpuNormed, wQ, 1, e.Config.Heads*e.Config.HeadDim, e.Config.Dim)

	wK := loadW(fmt.Sprintf("blk.%d.attn_k.weight", layerIdx))
	cpuK := device.CPUMatMul(cpuNormed, wK, 1, e.Config.KVHeads*e.Config.HeadDim, e.Config.Dim)

	wV := loadW(fmt.Sprintf("blk.%d.attn_v.weight", layerIdx))
	cpuV := device.CPUMatMul(cpuNormed, wV, 1, e.Config.KVHeads*e.Config.HeadDim, e.Config.Dim)

	// Since GPU Layer call above modified QPart/KPart in place with RoPE @ pos 0
	// We should first check them @ pos 0 where RoPE is identity.
	gpuQ_pos0 := scratch.QPart.ToHost()
	gpuK_pos0 := scratch.KPart.ToHost()
	gpuV := scratch.VPart.ToHost()

	assertClose("Q Projection", cpuQ, gpuQ_pos0, 0.05)
	fmt.Printf("Q Projection Sample: CPU %v GPU %v\n", cpuQ[:4], gpuQ_pos0[:4])
	assertClose("K Projection", cpuK, gpuK_pos0, 0.05)
	assertClose("V Projection", cpuV, gpuV, 0.05)

	// Now check RoPE at Pos 1
	// check RoPE at Pos 1 (it was applied during the first Layer call)
	gpuQ_RoPE := scratch.QPart.ToHost()
	// cpuQ already has both tokens? No, cpuQ was computed for rows.
	// We need cpuQ for rows 0 and 1.
	cpuQ_RoPE := device.CPURoPE(cpuQ, 0, e.Config.Heads, e.Config.HeadDim, e.Config.RopeTheta)
	assertClose("RoPE Q @ Pos 0/1", cpuQ_RoPE, gpuQ_RoPE, 1e-2)

	// Check Scores (Q*K)
	// GPU scores are in scratch.Scores (which is not copied to host yet)
	gpuScores := scratch.Scores.ToHost()
	
	// Q_pos1 is computed above as cpuQ_RoPE
	// K_pos0 and K_pos1:
	cpuK_Pos0 := device.CPURoPE(cpuK, 0, e.Config.KVHeads, e.Config.HeadDim, e.Config.RopeTheta)
	cpuK_Pos1 := device.CPURoPE(cpuK, 1, e.Config.KVHeads, e.Config.HeadDim, e.Config.RopeTheta)
	
	scale := 1.0 / float32(math.Sqrt(float64(e.Config.HeadDim)))
	
	// Compute manual dot product for Head 0
	// Head 0 Q is cpuQ_RoPE[0:HeadDim]
	// Head 0 K is cpuK_Pos0[0:HeadDim]
	
	// Dot 0 (Pos 1 vs Pos 0)
	var dot0 float32
	for i := 0; i < e.Config.HeadDim; i++ {
		dot0 += cpuQ_RoPE[i] * cpuK_Pos0[i]
	}
	dot0 *= scale
	
	// Dot 1 (Pos 1 vs Pos 1)
	var dot1 float32
	for i := 0; i < e.Config.HeadDim; i++ {
		dot1 += cpuQ_RoPE[i] * cpuK_Pos1[i]
	}
	dot1 *= scale
	
	// --- CPU Softmax ---
	maxScore := dot0
	if dot1 > maxScore { maxScore = dot1 }
	exp0 := float32(math.Exp(float64(dot0 - maxScore)))
	exp1 := float32(math.Exp(float64(dot1 - maxScore)))
	sumExp := exp0 + exp1
	prob0 := exp0 / sumExp
	prob1 := exp1 / sumExp
	
	fmt.Printf("Softmax Probabilities (CPU): [0] %f [1] %f\n", prob0, prob1)
	fmt.Printf("Attention Scores (GPU Probs): [0] %f [1] %f\n", gpuScores[0], gpuScores[1])
	
	diffProb0 := float32(math.Abs(float64(prob0 - gpuScores[0])))
	diffProb1 := float32(math.Abs(float64(prob1 - gpuScores[1])))
	
	if diffProb0 > 0.05 || diffProb1 > 0.05 {
		fmt.Printf("FAIL: Softmax Mismatch > 0.05\n")
	} else {
		fmt.Printf("PASS: Softmax Proabilities Match (Scores are correct).\n")
	}
	
	// --- Check Attention Values (Weighted Sum) ---
	// Need vCache
	gpuVCacheData := vCache.ToHost()
	// vCache shape: [debugSeqLen, KVHeads * HeadDim] or [debugSeqLen, KVHeads, HeadDim]?
	// Tensor is flat. 
	// Stride per row = KVHeads * HeadDim = 8 * 128 = 1024 floats.
	// Head 0 is first 128 elements of row.
	
	row0Offset := 0 * 1024
	row1Offset := 1 * 1024
	
	// Read V0 and V1 for Head 0
	v0 := gpuVCacheData[row0Offset : row0Offset+128]
	v1 := gpuVCacheData[row1Offset : row1Offset+128]
	
	// Compute Expected Weighted Sum for Head 0
	expectedOut := make([]float32, 128)
	for i := 0; i < 128; i++ {
		expectedOut[i] = prob0 * v0[i] + prob1 * v1[i]
	}
	
	// Read GPU AttOut
	// AttOut is [1, Heads * HeadDim] = [1, 4096]
	// Head 0 is first 128 elements.
	gpuAttOut := scratch.AttOut.ToHost() // This might be overwritten by O-Projection?
	// Wait. Layer code:
	// Output Projection: scratch.AttOut.LinearInto(...) -> scratch.ResAtt
	// So scratch.AttOut is INPUT to Linear. It should be preserved?
	// LinearInto reads w and src, writes dst.
	// scratch.AttOut is src.
	// So it should be preserved.
	
	gpuHead0Out := gpuAttOut[:128]
	
	fmt.Printf("AttOut Sample (First 5): CPU %v GPU %v\n", expectedOut[:5], gpuHead0Out[:5])
	assertClose("Attention Values (Head 0)", expectedOut, gpuHead0Out, 0.05)

	// --- Debug RoPE Frequencies ---
	fmt.Println("Debugging RoPE Frequencies...")
	freqBuf := e.Ctx.NewTensorFP32(1, e.Config.HeadDim/2)
	freqBuf.DebugRoPEFreq(e.Config.HeadDim, e.Config.RopeTheta, 1) // Pos 1
	e.Ctx.Synchronize()
	gpuFreq := freqBuf.ToHost()
	
	// CPU Freq
	cpuFreq := make([]float32, e.Config.HeadDim/2)
	for i := 0; i < e.Config.HeadDim/2; i++ {
		expVar := -2.0 * float32(i) / float32(e.Config.HeadDim)
		cpuFreq[i] = 1.0 * float32(math.Pow(float64(e.Config.RopeTheta), float64(expVar)))
	}
	
	fmt.Printf("RoPE Freq Sample (First 5): CPU %v GPU %v\n", cpuFreq[:5], gpuFreq[:5])
	fmt.Printf("RoPE Freq Sample (Last 5): CPU %v GPU %v\n", cpuFreq[59:], gpuFreq[59:]) // idx 0..63
	assertClose("RoPE Frequencies", cpuFreq, gpuFreq, 1e-3)

	// --- Check Dot Product of GPU Tensors ---
	gpuK_Pos0 := gpuK_pos0 // From line 130
	// gpuQ_RoPE is Q @ Pos 1
	
	var gpuDot0 float32
	for i := 0; i < e.Config.HeadDim; i++ {
		gpuDot0 += gpuQ_RoPE[i] * gpuK_Pos0[i]
	}
	gpuDot0 *= scale
	
	fmt.Printf("GPU Dot Product (Host Calc): %f\n", gpuDot0)
	fmt.Printf("GPU Attn Score (Kernel):     %f\n", gpuScores[0])
	
	diffGpu := float32(math.Abs(float64(gpuDot0 - gpuScores[0])))
	if diffGpu > 0.01 {
		fmt.Printf("FAIL: Kernel Logic Mismatch! Host-calculated dot product of GPU tensors %f != Kernel output %f\n", gpuDot0, gpuScores[0])
	} else {
		fmt.Printf("PASS: Kernel Logic Matches Host Dot Product. Mismatch is in Data values.\n")
	}
	
	// --- GPU Debug Dot ---
	fmt.Println("Running GPU Debug Dot Kernel...")
	debugDotBuf := e.Ctx.NewTensorFP32(1, 1)
	// Compute Dot(QPart, kCache) for first HeadDim elements
	scratch.QPart.DebugDot(kCache, debugDotBuf, e.Config.HeadDim)
	e.Ctx.Synchronize()
	gpuDebugDot := debugDotBuf.ToHost()[0]
	// debug_dot returns unscaled sum.
	// scale it
	gpuDebugDotScaled := gpuDebugDot * scale
	
	fmt.Printf("GPU Debug Dot (Kernel): %f (Scaled: %f)\n", gpuDebugDot, gpuDebugDotScaled)
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
