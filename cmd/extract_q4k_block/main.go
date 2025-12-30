package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"unsafe"
	
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model")
	tensorName := flag.String("tensor", "", "Tensor name to extract")
    blockIdx := flag.Int("block", 0, "Block index to read (for Quantized)")
	flag.Parse()

	if *modelPath == "" {
		*modelPath = "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
	}
	if *tensorName == "" {
		log.Fatal("Please provide --tensor name")
	}

	fmt.Printf("Loading model: %s\n", *modelPath)
	f, err := gguf.LoadFile(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load: %v", err)
	}
	
	var target *gguf.TensorInfo
	for _, t := range f.Tensors {
		if t.Name == *tensorName {
			target = t
			break
		}
	}
	if target == nil {
		log.Fatalf("Tensor %s not found", *tensorName)
	}
	
	fmt.Printf("Tensor: %s (Type: %d, Dims: %v)\n", target.Name, target.Type, target.Dimensions)
	
	if target.Type == 0 { // F32
		readF32(target, *blockIdx)
	} else if target.Type == 12 { // Q4_K
		readQ4K(target, *blockIdx)
	} else if target.Type == 14 { // Q6_K
		readQ6K(target, *blockIdx)
	} else {
		fmt.Printf("Unsupported type %d. Dumping raw hex of first 32 bytes.\n", target.Type)
		readRaw(target, 32)
	}
}

func readF32(t *gguf.TensorInfo, elementOffset int) {
    // Treat blockIdx as Element Offset for F32 (since no blocks)
    offset := uint64(elementOffset) * 4
    if offset+16 > uint64(len(t.Data)) {
        offset = 0 // Fallback
    }
	r := make([]byte, 16)
	copy(r, t.Data[offset : offset+16])
	for i := 0; i < 4; i++ {
		val := math.Float32frombits(binary.LittleEndian.Uint32(r[i*4 : (i+1)*4]))
		fmt.Printf("F32[%d]: %f\n", elementOffset+i, val)
	}
}

func readRaw(t *gguf.TensorInfo, n int) {
	r := make([]byte, n)
	copy(r, t.Data[0:n])
	fmt.Printf("Hex: %x\n", r)
}

func readQ4K(t *gguf.TensorInfo, blockIdx int) {
    // Block size 144 bytes
    offset := uint64(blockIdx) * 144
    if offset+144 > uint64(len(t.Data)) {
        log.Fatalf("Block index %d out of bounds", blockIdx)
    }
    r := make([]byte, 144) 
    copy(r, t.Data[offset : offset+144])
    d := float16ToFloat32(binary.LittleEndian.Uint16(r[0:2]))
    dmin := float16ToFloat32(binary.LittleEndian.Uint16(r[2:4]))
    fmt.Printf("Q4K Block %d: d=%f, dmin=%f (Offset %d)\n", blockIdx, d, dmin, offset)
}

func readQ6K(t *gguf.TensorInfo, blockIdx int) {
    // Block size 210 bytes
    offset := uint64(blockIdx) * 210
    if offset+210 > uint64(len(t.Data)) {
        log.Fatalf("Block index %d out of bounds", blockIdx)
    }
    r := make([]byte, 210)
    copy(r, t.Data[offset : offset+210])
    d := float16ToFloat32(binary.LittleEndian.Uint16(r[208:210]))
    fmt.Printf("Q6K Block %d: d=%f (Offset %d)\n", blockIdx, d, offset)
}

func float16ToFloat32(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	exp := uint32(bits&0x7C00) >> 10
	mant := uint32(bits & 0x03FF)
	if exp == 0 {
		if mant == 0 { return 0.0 }
		shift := uint32(0)
		for (mant & 0x0400) == 0 { mant <<= 1; shift++ }
		mant &= 0x03FF
		exp = 1 - shift
	}
	exp32 := (exp + (127 - 15)) << 23
	f32bits := sign | exp32 | (mant << 13)
	return *(*float32)(unsafe.Pointer(&f32bits))
}
