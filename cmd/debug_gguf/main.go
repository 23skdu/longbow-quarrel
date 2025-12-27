package main

import (
	"encoding/binary"
	"fmt"
	"os"
	
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: debug_gguf <path>")
		// Fallback to test gen if no args? Or just exit.
		// For now, let's keep the test gen if requested, or just use arg.
		// Let's make it mandatory.
		os.Exit(1)
	}
	filename := os.Args[1]
	
	// if err := generateGGUF(filename); err != nil {
	// 	panic(err)
	// }
	// defer os.Remove(filename)
	
	fmt.Println("Reading GGUF...")
	f, err := gguf.LoadFile(filename)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	
	fmt.Printf("Magic: %x\n", f.Header.Magic)
	fmt.Printf("Version: %d\n", f.Header.Version)
	fmt.Printf("Tensors: %d\n", f.Header.TensorCount)
	fmt.Printf("KV: %d\n", f.Header.KVCount)
	
	fmt.Println("Metadata:")
	for k, v := range f.KV {
		// Truncate long values for display
		s := fmt.Sprintf("%v", v)
		if len(s) > 100 {
			s = s[:97] + "..."
		}
		fmt.Printf("  %s: %s\n", k, s)
	}
	
	fmt.Println("Tensors:")
	for _, t := range f.Tensors {
		fmt.Printf("  %s: Dims %v, Type %d, Offset %d\n", t.Name, t.Dimensions, t.Type, t.Offset)
	}
}

func generateGGUF(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	
	// Magic
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMagic))
	// Version
	binary.Write(f, binary.LittleEndian, uint32(3))
	// Tensor Count (1)
	binary.Write(f, binary.LittleEndian, uint64(1))
	// KV Count (1)
	binary.Write(f, binary.LittleEndian, uint64(1))
	
	// KV Pair 1: "general.architecture" -> "llama"
	writeString(f, "general.architecture")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeString))
	writeString(f, "llama")
	
	// Tensor Info 1: "token_embd.weight"
	writeString(f, "token_embd.weight")
	// Dims (1)
	binary.Write(f, binary.LittleEndian, uint32(1))
	// Ne[0] = 1
	binary.Write(f, binary.LittleEndian, uint64(1))
	// Type = F32
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGMLTypeF32))
	// Offset = 0
	binary.Write(f, binary.LittleEndian, uint64(0))
	
	// Alignment padding
	// Current offset? 
	// Magic(4)+Ver(4)+TC(8)+KC(8) = 24
	// KV: len(8)+"general.architecture"(20)+type(4)+len(8)+"llama"(5) = 45
	// Total so far = 69
	// TensorInfo:
	// len(8)+"token_embd.weight"(17) = 25
	// dims(4) + ne(8) + type(4) + off(8) = 24
	// Total info = 49
	// Total header = 69 + 49 = 118
	
	// Alignment 32. Next multiple of 32 after 118 is 128.
	// Padding = 10 bytes
	pad := make([]byte, 10)
	f.Write(pad)
	
	// Tensor Data (4 bytes for 1 float32)
	// 123.456
	binary.Write(f, binary.LittleEndian, float32(123.456))
	
	return nil
}

func writeString(f *os.File, s string) {
	binary.Write(f, binary.LittleEndian, uint64(len(s)))
	f.WriteString(s)
}
