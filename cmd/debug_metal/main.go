//go:build darwin && metal

package main

import (
	"fmt"
	"math"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

func main() {
	println("Initializing Metal Context...")
	ctx := device.NewContext()
	defer ctx.Free()

	testAdd(ctx)
	testScale(ctx)
	testMatMul(ctx)
	testRMSNorm(ctx)
}

func testAdd(ctx *device.Context) {
	fmt.Println("Testing Add Kernel...")
	rows, cols := 10, 10
	size := rows * cols

	aData := make([]float32, size)
	bData := make([]float32, size)
	expected := make([]float32, size)

	for i := 0; i < size; i++ {
		aData[i] = float32(i)
		bData[i] = float32(i * 2)
		expected[i] = aData[i] + bData[i]
	}

	tA := ctx.NewTensor(rows, cols)
	tA.LoadFrom(aData)

	tB := ctx.NewTensor(rows, cols)
	tB.LoadFrom(bData)

	tC := tA.Add(tB)
	result := tC.ToHost()

	if !almostEqual(result, expected) {
		panic(fmt.Sprintf("Add failed. Expected vs Result mismatch"))
	}
	fmt.Println("Add Passed!")
}

func testScale(ctx *device.Context) {
	fmt.Println("Testing Scale Kernel...")
	rows, cols := 5, 5
	size := rows * cols
	
	aData := make([]float32, size)
	scale := float32(2.5)
	expected := make([]float32, size)
	
	for i := 0; i < size; i++ {
		aData[i] = float32(i)
		expected[i] = aData[i] * scale
	}
	
	tA := ctx.NewTensor(rows, cols)
	tA.LoadFrom(aData)
	
	tC := tA.ScaleBy(scale)
	result := tC.ToHost()
	
	if !almostEqual(result, expected) {
		panic("Scale failed")
	}
	fmt.Println("Scale Passed!")
}

func testMatMul(ctx *device.Context) {
    // A: 2x4, B: 4x2, C: 2x2
    fmt.Println("Testing MatMul...")
    M, K, N := 2, 4, 2
    
    aData := []float32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    }
    bData := []float32{
        1, 0,
        0, 1,
        1, 0,
        0, 1,
    }
    // Result should be:
    // [1*1 + 2*0 + 3*1 + 4*0,  1*0 + 2*1 + 3*0 + 4*1] = [1+3, 2+4] = [4, 6]
    // [5*1 + 6*0 + 7*1 + 8*0,  5*0 + 6*1 + 7*0 + 8*1] = [5+7, 6+8] = [12, 14]
    
    tA := ctx.NewTensor(M, K)
    tA.LoadFrom(aData)
    
    tB := ctx.NewTensor(K, N)
    tB.LoadFrom(bData)
    
    tC := tA.MatMul(tB)
    
    result := tC.ToHost()
    // fmt.Printf("Result: %v\n", result)
    
    expected := []float32{4, 6, 12, 14}
    if !almostEqual(result, expected) {
        panic(fmt.Sprintf("MatMul failed. Expected %v, got %v", expected, result))
    }
    fmt.Println("MatMul Passed!")
}

func testRMSNorm(ctx *device.Context) {
    fmt.Println("Testing RMSNorm...")
    // 2 rows, 4 cols
    rows, cols := 2, 4
    // Row 1: 1,1,1,1 -> RMS=1 -> Out=1
    // Row 2: 2,2,2,2 -> RMS=2 -> Out=1
    input := []float32{
        1, 1, 1, 1, 
        2, 2, 2, 2, 
    }
    weight := []float32{1, 1, 1, 1}
    
    tIn := ctx.NewTensor(rows, cols)
    tIn.LoadFrom(input)
    
    tW := ctx.NewTensor(1, cols)
    tW.LoadFrom(weight)
    
    res := tIn.RMSNorm(tW, 1e-5)
    out := res.ToHost()
    
    // fmt.Printf("RMSNorm Result: %v\n", out)
    for _, v := range out {
    	if math.Abs(float64(v - 1.0)) > 1e-2 {
    		panic(fmt.Sprintf("RMSNorm failed. Expected ~1.0, got %v", v))
    	}
    }
    fmt.Println("RMSNorm Passed!")
}

func almostEqual(a, b []float32) bool {
    if len(a) != len(b) { return false }
    for i := range a {
        diff := a[i] - b[i]
        if diff < 0 { diff = -diff }
        // FP16 precision is low, be generous
        if diff > 1e-1 { 
        	// Check relative error for larger numbers
        	if b[i] != 0 {
        		rel := diff / b[i]
        		if rel < 0 { rel = -rel }
        		if rel > 0.05 { return false }
        	} else {
        		return false 
        	}
        }
    }
    return true
}
