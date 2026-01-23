package main

import (
	"encoding/binary"
	"os"
)

// Simplified generator
func main() {
	f, _ := os.Create("test.gguf")
	defer f.Close()

	// Magic 0x46554747
	binary.Write(f, binary.LittleEndian, uint32(0x46554747))
	// Version 3
	binary.Write(f, binary.LittleEndian, uint32(3))
	// Tensor Count 1
	binary.Write(f, binary.LittleEndian, uint64(1))
	// KV Count 0
	binary.Write(f, binary.LittleEndian, uint64(0))

	// Tensor Info: "A"
	binary.Write(f, binary.LittleEndian, uint64(1))
	f.WriteString("A")
	// Dims 1
	binary.Write(f, binary.LittleEndian, uint32(1))
	// Ne[0] = 1
	binary.Write(f, binary.LittleEndian, uint64(1))
	// Type F32 (0)
	binary.Write(f, binary.LittleEndian, uint32(0))
	// Offset 0
	binary.Write(f, binary.LittleEndian, uint64(0))

	// Padding (alignment 32).
	// Header: 24 bytes
	// TensorInfo: 8+1(len)+1(str)=10. +4+8+4+8 = 24.
	// Total = 48. Next 32 is 64. Pad 16.
	f.Write(make([]byte, 16))

	// Data: 1.0 (float32)
	binary.Write(f, binary.LittleEndian, float32(1.0))
}
