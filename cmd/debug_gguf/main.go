//go:build darwin && metal

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"syscall"
)

const (
	GGUFMagic = 0x46554747
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: check_gguf <path>")
		return
	}
	path := os.Args[1]
	
	f, err := os.Open(path)
	if err != nil { panic(err) }
	defer f.Close()
	
	info, _ := f.Stat()
	size := info.Size()
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil { panic(err) }
	defer syscall.Munmap(data)
	
	offset := uint64(0)
	magic := binary.LittleEndian.Uint32(data[offset:])
	offset += 4
	fmt.Printf("Magic: %x\n", magic)
	
	version := binary.LittleEndian.Uint32(data[offset:])
	offset += 4
	fmt.Printf("Version: %d\n", version)
	
	tensorCount := binary.LittleEndian.Uint64(data[offset:])
	offset += 8
	kvCount := binary.LittleEndian.Uint64(data[offset:])
	offset += 8
	
	fmt.Printf("Tensors: %d, KV: %d\n", tensorCount, kvCount)
	
	// Skip KV
	fmt.Printf("Starting KV read at offset %d\n", offset)
	for i := uint64(0); i < kvCount; i++ {
		k, n := readString(data, offset)
		offset += n
		valType := binary.LittleEndian.Uint32(data[offset:])
		offset += 4
		
		if valType == 4 { // Uint32
			v := binary.LittleEndian.Uint32(data[offset:])
			fmt.Printf("KV %d: Key=%s, Type=%d, Val=%d\n", i, k, valType, v)
		} else if valType == 6 { // Float32
			bits := binary.LittleEndian.Uint32(data[offset:])
			fmt.Printf("KV %d: Key=%s, Type=%d, Val=%f\n", i, k, valType, math.Float32frombits(bits))
		} else {
			fmt.Printf("KV %d: Key=%s, Type=%d\n", i, k, valType)
		}

		_, n = readValue(data, offset, valType)
		offset += n
	}
	
	fmt.Printf("Offset after KV: %d\n", offset)
	
	// Read Tens
	type Ten struct {
		Name string
		Off uint64
		Size uint64
	}
	var lastTen Ten
	
	for i := uint64(0); i < tensorCount; i++ {
		name, n := readString(data, offset)
		offset += n
		dims := binary.LittleEndian.Uint32(data[offset:])
		offset += 4
		
		elements := uint64(1)
		for j := uint32(0); j < dims; j++ {
			d := binary.LittleEndian.Uint64(data[offset:])
			elements *= d
			offset += 8
		}
		
		typ := binary.LittleEndian.Uint32(data[offset:])
		offset += 4
		tenOff := binary.LittleEndian.Uint64(data[offset:])
		offset += 8
		
		// Approx size
		// Q4_K(12): 144 bytes / 256
		// F16(1): 2 bytes
		// F32(0): 4 bytes
		sz := uint64(0)
		if typ == 12 {
			sz = (elements / 256) * 144
		} else if typ == 1 {
			sz = elements * 2
		} else if typ == 0 {
			sz = elements * 4
		} else if typ == 14 { // Q6K
			sz = (elements / 256) * 210
		}
		
            
            // Print all tensors
			fmt.Printf("Tensor %d: %s: Off=%d, Elements=%d, Typ=%d, EstSize=%d\n", i, name, tenOff, elements, typ, sz)
		
		lastTen = Ten{Name: name, Off: tenOff, Size: sz}
	}
	
	fmt.Printf("Offset after TensorInfos: %d\n", offset)
	
	padding := uint64(32) - (offset % 32)
	if padding == 32 { padding = 0 }
	fmt.Printf("Padding: %d -> DataStart: %d\n", padding, offset + padding)
	
	// Check last tensor end
	end := offset + padding + lastTen.Off + lastTen.Size
	fmt.Printf("Last Tensor: %s, End: %d, FileSize: %d, Diff: %d\n", lastTen.Name, end, size, int64(size)-int64(end))
}

func readString(data []byte, offset uint64) (string, uint64) {
	l := binary.LittleEndian.Uint64(data[offset:])
    fmt.Printf("READSTRING: Off=%d, Len=%d\n", offset, l)
	return string(data[offset+8 : offset+8+l]), 8+l
}

func readValue(data []byte, offset uint64, typ uint32) (interface{}, uint64) {
	// Minimal skipper
	switch typ {
	case 8: // string
		_, n := readString(data, offset)
		return nil, n
	case 9: // array
		// type, len, elements
		atype := binary.LittleEndian.Uint32(data[offset:])
		alen := binary.LittleEndian.Uint64(data[offset+4:])
		n := uint64(12)
		off := offset + 12
		for i:=uint64(0); i<alen; i++ {
			_, sn := readValue(data, off, atype)
			n += sn
			off += sn
		}
		return nil, n
	default:
		// primitive fixed sizes?
		// bool(10)=1, u8(0)=1, i8(1)=1, u16(2)=2, i16(3)=2, u32(4)=4, i32(5)=4, f32(6)=4, u64(7)=8, i64(11)=8, f64(12)=8
		sz := uint64(0)
		switch typ {
		case 0,1,7: sz=1 // Bool is 7
		case 2,3: sz=2
		case 4,5,6: sz=4
		case 10,11,12: sz=8
		}
		return nil, sz
	}
}
