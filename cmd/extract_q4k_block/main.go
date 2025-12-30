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
	
	// Find tensor
	tensorName := "output_norm.weight"
	tensorIdx := -1
	for i, info := range f.Tensors {
		if info.Name == tensorName {
			tensorIdx = i
			break
		}
	}
	
	var targetTensor *gguf.TensorInfo
	if tensorIdx != -1 {
		targetTensor = f.Tensors[tensorIdx]
	}

	if targetTensor == nil {
		log.Fatalf("Could not find %s", tensorName)
	}
	
	fmt.Printf("Found tensor: %s\n", targetTensor.Name)
	fmt.Printf("  Type: %d (Q4_K)\n", targetTensor.Type)
	fmt.Printf("  Dimensions: %v\n", targetTensor.Dimensions)
	// Calculate offset for Token 1782 from output.weight
	// output.weight is [4096, 32768].
	// We want row 1782.
	// Row size = 4096 elements.
	// Q6K: 256 elements per block. Size 210 bytes.
	// Blocks per row = 4096 / 256 = 16 blocks.
	// Row bytes = 16 * 210 = 3360 bytes.
	// Offset = 1782 * 3360.
	
	byteOffset := uint64(1782) * 16 * 210
	
	fmt.Printf("Extracting Q6K block %d for Token 1782 (Byte Offset %d)...\n", 1782*16, byteOffset)
	
	blockLen := 210
	
	if byteOffset+uint64(blockLen) > uint64(len(targetTensor.Data)) {
		log.Fatalf("Offset out of bounds")
	}
	
	blockData := targetTensor.Data[byteOffset : byteOffset+uint64(blockLen)]
	
	// Q6K Structure:
	// ql: 128 bytes
	// qh: 64 bytes
	// scales: 16 bytes
	// d: 2 bytes (FP16)
	
	d_bits := binary.LittleEndian.Uint16(blockData[208:210])
	d := float16ToFloat32(d_bits)
	
	fmt.Printf("\nFirst Q6K Block Analysis:\n")
	fmt.Printf("  d (scale):     %.12f (bits: %04x)\n", d, d_bits)
	
	ql := blockData[0:128]
	qh := blockData[128:192]
	scales := blockData[192:208]
	
	fmt.Printf("  ql[0:4]: %v\n", ql[:4])
	fmt.Printf("  qh[0:4]: %v\n", qh[:4])
	fmt.Printf("  sc[0:4]: %v\n", scales[:4])
	
	// Save to file
	err = os.WriteFile("token_the_q6k_block.bin", blockData, 0644)
	if err != nil {
		log.Printf("Error writing file: %v", err)
	} else {
		fmt.Println("Saved Q6K block to token_the_q6k_block.bin")
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
