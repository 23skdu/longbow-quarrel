package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"unsafe"
	
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func main() {
	modelPath := "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
	
	fmt.Printf("Loading model: %s\n", modelPath)
	f, err := gguf.LoadFile(modelPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	
	// Find blk.0.attn_q.weight
	var targetTensor *gguf.TensorInfo
	for _, t := range f.Tensors {
		if t.Name == "blk.0.attn_q.weight" {
			targetTensor = t
			break
		}
	}
	
	if targetTensor == nil {
		log.Fatal("Could not find blk.0.attn_q.weight")
	}
	
	fmt.Printf("Found tensor: %s\n", targetTensor.Name)
	fmt.Printf("  Type: %d (Q4_K)\n", targetTensor.Type)
	fmt.Printf("  Dimensions: %v\n", targetTensor.Dimensions)
	fmt.Printf("  Data size: %d bytes\n", len(targetTensor.Data))
	
	// Extract first Q4K block (144 bytes)
	if len(targetTensor.Data) < 144 {
		log.Fatalf("Tensor data too small: %d bytes", len(targetTensor.Data))
	}
	
	block := targetTensor.Data[0:144]
	
	// Parse and display block info
	d_bits := binary.LittleEndian.Uint16(block[0:2])
	dmin_bits := binary.LittleEndian.Uint16(block[2:4])
	
	// Convert FP16 to FP32 (simple, may not handle subnormals)
	d := float16ToFloat32(d_bits)
	dmin := float16ToFloat32(dmin_bits)
	
	fmt.Printf("\nFirst Q4K Block Analysis:\n")
	fmt.Printf("  d (scale):     %.12f (bits: %04x)\n", d, d_bits)
	fmt.Printf("  dmin (offset): %.12f (bits: %04x)\n", dmin, dmin_bits)
	
	// Show first few scale bytes
	fmt.Printf("  scales[0:4]: %v\n", block[4:8])
	fmt.Printf("  quants[0:4]: %v\n", block[16:20])
	
	// Save block to file
	outputPath := "mistral_q4k_block_0.bin"
	err = os.WriteFile(outputPath, block, 0644)
	if err != nil {
		log.Fatalf("Failed to write block: %v", err)
	}
	
	fmt.Printf("\nSaved first Q4K block to: %s\n", outputPath)
	
	// Also extract a few more blocks for comparison
	for i := 1; i < 5; i++ {
		if len(targetTensor.Data) >= (i+1)*144 {
			blockN := targetTensor.Data[i*144 : (i+1)*144]
			d_n := float16ToFloat32(binary.LittleEndian.Uint16(blockN[0:2]))
			dmin_n := float16ToFloat32(binary.LittleEndian.Uint16(blockN[2:4]))
			fmt.Printf("Block %d: d=%.12f, dmin=%.12f\n", i, d_n, dmin_n)
		}
	}
}

func float16ToFloat32(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	exp := uint32(bits&0x7C00) >> 10
	mant := uint32(bits & 0x03FF)
	
	if exp == 0 {
		if mant == 0 {
			return 0.0
		}
		// Subnormal - normalize it
		shift := uint32(0)
		for (mant & 0x0400) == 0 {
			mant <<= 1
			shift++
		}
		mant &= 0x03FF
		exp = 1 - shift
	}
	
	exp32 := (exp + (127 - 15)) << 23
	f32bits := sign | exp32 | (mant << 13)
	
	return *(*float32)(unsafe.Pointer(&f32bits))
}
