package gguf

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"syscall"
	"unsafe"
)

// LoadFile maps a GGUF file into memory and parses headers/metadata.
func LoadFile(path string) (*GGUFFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = f.Close() // Ignore close error in reader function
	}()

	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := info.Size()

	// Memory map the file
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap failed: %w", err)
	}

	file := &GGUFFile{
		Data: data,
		KV:   make(map[string]interface{}),
	}

	offset := uint64(0)

	// Read Header
	if size < 24 { // Minimal header size check
		return nil, io.ErrUnexpectedEOF
	}

	file.Header.Magic = binary.LittleEndian.Uint32(data[offset:])
	offset += 4
	if file.Header.Magic != GGUFMagic {
		return nil, ErrInvalidMagic{Magic: file.Header.Magic}
	}

	file.Header.Version = binary.LittleEndian.Uint32(data[offset:])
	offset += 4

	// We support version 2 and 3 mostly
	if file.Header.Version < 2 || file.Header.Version > 3 {
		return nil, ErrUnsupportedVersion{Version: file.Header.Version}
	}

	file.Header.TensorCount = binary.LittleEndian.Uint64(data[offset:])
	offset += 8
	file.Header.KVCount = binary.LittleEndian.Uint64(data[offset:])
	offset += 8

	fmt.Fprintf(os.Stderr, "GGUF: Version=%d, Tensors=%d, KV=%d\n", file.Header.Version, file.Header.TensorCount, file.Header.KVCount)

	// Ollama format stores KV pairs at the END of the file, not at the beginning
	// If KVCount is 0, search for tokenizer data at the end
	if file.Header.KVCount == 0 {
		// Search for "tokenizer.ggml.tokens" in the file
		tokenizerIdx := bytes.Index(data, []byte("tokenizer.ggml.tokens"))
		if tokenizerIdx > 0 {
			fmt.Fprintf(os.Stderr, "DEBUG: Found ollama format with tokenizer at offset %d\n", tokenizerIdx)
			// Parse KV pairs from the end of the file
			// Search backwards from tokenizer to find the KV section
			// The format is: [string key][uint32 type][value...] repeated
			// Starting from tokenizer.ggml.tokens

			pos := uint64(tokenizerIdx)
			for pos > 0 {
				// Look for a string key before current position
				// Keys are prefixed with length (uint64)
				keyStart := uint64(0)
				for i := int(pos - 8); i >= 0 && i > int(pos-200); i-- {
					if data[i] == 0 && i+8 < len(data) {
						// Check if this could be a string length prefix
						strLen := binary.LittleEndian.Uint64(data[i:])
						if strLen > 0 && strLen < 100 && i+8+int(strLen) < int(pos) {
							keyStart = uint64(i + 8)
							break
						}
					}
				}

				if keyStart == 0 {
					break
				}

				// Read the key
				keyLen := binary.LittleEndian.Uint64(data[keyStart-8:])
				if keyStart+keyLen >= pos {
					break
				}

				key := string(data[keyStart : keyStart+keyLen])
				keyOff := keyStart - 8

				// Read value type
				valType := GGUFMetadataValueType(binary.LittleEndian.Uint32(data[keyOff+keyLen:]))
				valOff := keyOff + keyLen + 4

				// Read value
				val, _, err := readValue(data, valOff, valType)
				if err != nil {
					break
				}

				file.KV[key] = val
				fmt.Fprintf(os.Stderr, "DEBUG: Loaded KV: %s\n", key)

				// Move back to find more keys
				pos = keyOff - 8

				// Stop if we've gone too far back
				if pos < 100 {
					break
				}
			}
		}
	} else {
		// Standard GGUF format - read KV pairs from beginning
		for i := uint64(0); i < file.Header.KVCount; i++ {
			k, n, err := readString(data, offset)
			if err != nil {
				return nil, err
			}
			offset += n

			valType := GGUFMetadataValueType(binary.LittleEndian.Uint32(data[offset:]))
			offset += 4

			val, n, err := readValue(data, offset, valType)
			if err != nil {
				return nil, err
			}
			offset += n

			file.KV[k] = val
		}
	}

	// Read Tensor Infos
	for i := uint64(0); i < file.Header.TensorCount; i++ {
		name, n, err := readString(data, offset)
		if err != nil {
			return nil, err
		}
		offset += n

		dims := binary.LittleEndian.Uint32(data[offset:])
		offset += 4

		dimArr := make([]uint64, dims)
		for j := uint32(0); j < dims; j++ {
			dimArr[j] = binary.LittleEndian.Uint64(data[offset:])
			offset += 8
		}

		typ := GGMLType(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4

		tensorOffset := binary.LittleEndian.Uint64(data[offset:])
		offset += 8

		file.Tensors = append(file.Tensors, &TensorInfo{
			Name:       name,
			Dimensions: dimArr,
			Type:       typ,
			Offset:     tensorOffset,
		})
		fmt.Fprintf(os.Stderr, "GGUF: Found tensor %s (Type: %v, Dims: %v)\n", name, typ, dimArr)
	}

	// Align to 32 bytes (GGUF alignment) before data starts
	// Actually, GGUF spec says alignment is given in KV pairs usually, typically 32.
	// We need to calculate data start.
	// The offsets in TensorInfo are relative to the end of the header block + alignment padding.

	// Default alignment
	alignment := uint64(32)
	if alignVal, ok := file.KV["general.alignment"].(uint32); ok {
		alignment = uint64(alignVal)
	} else if alignVal, ok := file.KV["general.alignment"].(uint64); ok { // sometimes it's u64?
		alignment = alignVal
	} else if alignVal, ok := file.KV["general.alignment"].(float64); ok { // JSON parser weirdness
		alignment = uint64(alignVal)
	}

	fmt.Fprintf(os.Stderr, "DEBUG: Alignment = %d\n", alignment)
	fmt.Fprintf(os.Stderr, "DEBUG: Offset BEFORE padding: %d\n", offset)

	// Pad offset to alignment
	padding := alignment - (offset % alignment)
	if padding != alignment {
		offset += padding
	}

	file.DataOffset = offset
	fmt.Fprintf(os.Stderr, "DEBUG: Computed Padding Offset: %d\n", offset)

	// Update tensor pointers
	fmt.Fprintf(os.Stderr, "DEBUG: Data Start Offset: %d\n", offset)
	for _, t := range file.Tensors {
		// Absolute offset = dataStart + t.Offset
		absOffset := offset + t.Offset
		if absOffset >= uint64(len(data)) {
			return nil, fmt.Errorf("tensor offset out of bounds")
		}

		t.Data = data[absOffset:]
	}

	return file, nil
}

func readString(data []byte, offset uint64) (string, uint64, error) {
	if offset+8 > uint64(len(data)) {
		return "", 0, io.ErrUnexpectedEOF
	}
	length := binary.LittleEndian.Uint64(data[offset:])

	if offset+8+length > uint64(len(data)) {
		return "", 0, io.ErrUnexpectedEOF
	}

	// Zero-copy string? unsafe?
	// str := unsafe.String(&data[offset+8], length)
	// Safe way for now to ensure it works
	str := string(data[offset+8 : offset+8+length])

	return str, 8 + length, nil
}

func readValue(data []byte, offset uint64, typ GGUFMetadataValueType) (interface{}, uint64, error) {
	switch typ {
	case GGUFMetadataValueTypeUint8:
		return data[offset], 1, nil
	case GGUFMetadataValueTypeInt8:
		return int8(data[offset]), 1, nil
	case GGUFMetadataValueTypeUint16:
		return binary.LittleEndian.Uint16(data[offset:]), 2, nil
	case GGUFMetadataValueTypeInt16:
		return int16(binary.LittleEndian.Uint16(data[offset:])), 2, nil
	case GGUFMetadataValueTypeUint32:
		return binary.LittleEndian.Uint32(data[offset:]), 4, nil
	case GGUFMetadataValueTypeInt32:
		return int32(binary.LittleEndian.Uint32(data[offset:])), 4, nil
	case GGUFMetadataValueTypeFloat32:
		return math_Float32frombits(binary.LittleEndian.Uint32(data[offset:])), 4, nil
	case GGUFMetadataValueTypeBool:
		return data[offset] != 0, 1, nil
	case GGUFMetadataValueTypeString:
		return readString(data, offset)
	case GGUFMetadataValueTypeArray:
		// Read type, then len, then elements
		arrType := GGUFMetadataValueType(binary.LittleEndian.Uint32(data[offset:]))
		arrLen := binary.LittleEndian.Uint64(data[offset+4:])
		bytesRead := uint64(12)
		currentOff := offset + 12

		var arr []interface{}
		for i := uint64(0); i < arrLen; i++ {
			val, n, err := readValue(data, currentOff, arrType)
			if err != nil {
				return nil, 0, err
			}
			arr = append(arr, val)
			currentOff += n
			bytesRead += n
		}
		return arr, bytesRead, nil
	case GGUFMetadataValueTypeUint64:
		return binary.LittleEndian.Uint64(data[offset:]), 8, nil
	case GGUFMetadataValueTypeInt64:
		return int64(binary.LittleEndian.Uint64(data[offset:])), 8, nil
	case GGUFMetadataValueTypeFloat64:
		return math.Float64frombits(binary.LittleEndian.Uint64(data[offset:])), 8, nil
	default:
		return nil, 0, fmt.Errorf("unsupported metadata type: %d", typ)
	}
}

// Helper to avoid math import if possible, but we need it for Float32frombits
// We can just use unsafe or define it.
// Actually we can import math
func math_Float32frombits(b uint32) float32 {
	return *(*float32)(unsafe.Pointer(&b))
}

func (f *GGUFFile) Close() error {
	return syscall.Munmap(f.Data)
}

func (f *GGUFFile) GetGapTensors() []*TensorInfo {
	// Sort tensors by offset
	sortedTensors := make([]*TensorInfo, len(f.Tensors))
	copy(sortedTensors, f.Tensors)
	sort.Slice(sortedTensors, func(i, j int) bool {
		return sortedTensors[i].Offset < sortedTensors[j].Offset
	})

	var gaps []*TensorInfo
	for i := 0; i < len(sortedTensors)-1; i++ {
		curr := sortedTensors[i]
		next := sortedTensors[i+1]

		currEnd := curr.Offset + curr.SizeBytes()

		if currEnd < next.Offset {
			gapSize := next.Offset - currEnd
			if gapSize > 1024*1024 { // Only care about large gaps (>1MB)
				absStart := f.DataOffset + currEnd
				absEnd := f.DataOffset + next.Offset
				gaps = append(gaps, &TensorInfo{
					Name:   fmt.Sprintf("gap_%d_at_%d", len(gaps), currEnd),
					Offset: currEnd,
					Data:   f.Data[absStart:absEnd],
				})
			}
		}
	}
	return gaps
}
