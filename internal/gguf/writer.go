package gguf

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
)

// The repository could read GGUF files but not write them, so the
// quantization functions in this package had nowhere to put their output:
// there was no way to turn a model into a quantized model.

// GGUFDefaultAlignment is the alignment the GGUF specification uses when
// the file does not say otherwise.
const GGUFDefaultAlignment = 32

// TensorData is a tensor to be written to a GGUF file.
type TensorData struct {
	Name       string
	Dimensions []uint64
	Type       GGMLType
	Data       []byte
}

// SizeBytes returns how many bytes the tensor's data occupies for its type
// and dimensions.
func (t TensorData) SizeBytes() uint64 {
	numElements := uint64(1)
	for _, d := range t.Dimensions {
		numElements *= d
	}

	switch t.Type {
	case GGMLTypeF32:
		return numElements * 4
	case GGMLTypeF16, GGMLTypeBF16:
		return numElements * 2
	case GGMLTypeQ4_0:
		return (numElements / 32) * 18
	case GGMLTypeQ5_0:
		return (numElements / 32) * 22
	case GGMLTypeQ8_0:
		return (numElements / 32) * 34
	case GGMLTypeQ4_K:
		return (numElements / 256) * 144
	case GGMLTypeQ6_K:
		return (numElements / 256) * 210
	case GGMLTypeQ3_K:
		return (numElements / 256) * 110
	case GGMLTypeQ2_K:
		return (numElements / 256) * 84
	case GGMLTypeIQ4_NL:
		return (numElements / 32) * 18
	case GGMLTypeTQ1_0, GGMLTypeTQ2_0:
		return (numElements / 256) * 328
	default:
		return 0
	}
}

// WriteFile writes a GGUF version 3 file.
//
// kv holds the metadata. The keys are written in sorted order, so that the
// same input always produces the same bytes. An array value may be any
// slice of a supported element type; the elements must all have the same
// type, because the GGUF array type is per-array.
//
// alignment is the tensor data alignment. Zero means the GGUF default of
// 32 bytes.
func WriteFile(path string, kv map[string]interface{}, tensors []TensorData, alignment uint64) error {
	if alignment == 0 {
		alignment = GGUFDefaultAlignment
	}

	body, err := buildGGUF(kv, tensors, alignment)
	if err != nil {
		return err
	}

	dir := filepath.Dir(path)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0750); err != nil {
			return fmt.Errorf("failed to create the output directory: %w", err)
		}
	}

	// Write to a temporary file and rename it, so that an interrupted
	// write does not leave a half-written model behind.
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, body, 0600); err != nil {
		return fmt.Errorf("failed to write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("failed to rename %s to %s: %w", tmp, path, err)
	}
	return nil
}

// buildGGUF assembles the whole file in memory.
func buildGGUF(kv map[string]interface{}, tensors []TensorData, alignment uint64) ([]byte, error) {
	var buf bytes.Buffer

	// Header.
	writeU32(&buf, GGUFMagic)
	writeU32(&buf, 3) // version
	writeU64(&buf, uint64(len(tensors)))
	writeU64(&buf, uint64(len(kv)))

	// Metadata, in sorted key order so the output is reproducible.
	keys := make([]string, 0, len(kv))
	for k := range kv {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		writeString(&buf, k)
		if err := writeValue(&buf, kv[k]); err != nil {
			return nil, fmt.Errorf("failed to write the metadata key %q: %w", k, err)
		}
	}

	// The tensor infos are built separately, because the offset they
	// carry is relative to the start of the tensor data, and the size of
	// the data section is not known until the infos are laid out.
	var infoBuf bytes.Buffer
	offsetPositions := make([]int, len(tensors))

	for i, t := range tensors {
		writeString(&infoBuf, t.Name)
		writeU32(&infoBuf, uint32(len(t.Dimensions)))
		for _, d := range t.Dimensions {
			writeU64(&infoBuf, d)
		}
		writeU32(&infoBuf, uint32(t.Type))
		offsetPositions[i] = infoBuf.Len()
		writeU64(&infoBuf, 0) // placeholder
	}

	// The data section starts after the header and the infos, aligned.
	dataStart := uint64(buf.Len() + infoBuf.Len())
	padding := alignment - (dataStart % alignment)
	if padding != alignment {
		infoBuf.Write(make([]byte, padding))
	}

	// Lay the tensors out and patch their offsets into the infos.
	offset := uint64(0)
	for i, t := range tensors {
		want := t.SizeBytes()
		if uint64(len(t.Data)) != want {
			return nil, fmt.Errorf("the tensor %q holds %d bytes, but its type and dimensions need %d",
				t.Name, len(t.Data), want)
		}
		binary.LittleEndian.PutUint64(infoBuf.Bytes()[offsetPositions[i]:], offset)
		offset += want
	}

	buf.Write(infoBuf.Bytes())

	// The tensor data itself.
	for _, t := range tensors {
		buf.Write(t.Data)
	}

	return buf.Bytes(), nil
}

func writeU32(buf *bytes.Buffer, v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	buf.Write(b[:])
}

func writeU64(buf *bytes.Buffer, v uint64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	buf.Write(b[:])
}

func writeString(buf *bytes.Buffer, s string) {
	writeU64(buf, uint64(len(s)))
	buf.WriteString(s)
}

// writeValue writes a metadata value, prefixed by nothing: the caller has
// already written the type.
func writeValue(buf *bytes.Buffer, v interface{}) error {
	switch t := v.(type) {
	case uint8:
		writeU32(buf, uint32(GGUFMetadataValueTypeUint8))
		buf.WriteByte(t)
	case int8:
		writeU32(buf, uint32(GGUFMetadataValueTypeInt8))
		buf.WriteByte(byte(t))
	case uint16:
		writeU32(buf, uint32(GGUFMetadataValueTypeUint16))
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], t)
		buf.Write(b[:])
	case int16:
		writeU32(buf, uint32(GGUFMetadataValueTypeInt16))
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], uint16(t))
		buf.Write(b[:])
	case uint32:
		writeU32(buf, uint32(GGUFMetadataValueTypeUint32))
		writeU32(buf, t)
	case int32:
		writeU32(buf, uint32(GGUFMetadataValueTypeInt32))
		writeU32(buf, uint32(t))
	case uint64:
		writeU32(buf, uint32(GGUFMetadataValueTypeUint64))
		writeU64(buf, t)
	case int64:
		writeU32(buf, uint32(GGUFMetadataValueTypeInt64))
		writeU64(buf, uint64(t))
	case float32:
		writeU32(buf, uint32(GGUFMetadataValueTypeFloat32))
		writeU32(buf, math.Float32bits(t))
	case float64:
		writeU32(buf, uint32(GGUFMetadataValueTypeFloat64))
		writeU64(buf, math.Float64bits(t))
	case bool:
		writeU32(buf, uint32(GGUFMetadataValueTypeBool))
		if t {
			buf.WriteByte(1)
		} else {
			buf.WriteByte(0)
		}
	case string:
		writeU32(buf, uint32(GGUFMetadataValueTypeString))
		writeString(buf, t)
	case []string:
		writeU32(buf, uint32(GGUFMetadataValueTypeArray))
		writeU32(buf, uint32(GGUFMetadataValueTypeString))
		writeU64(buf, uint64(len(t)))
		for _, s := range t {
			writeString(buf, s)
		}
	case []uint32:
		writeU32(buf, uint32(GGUFMetadataValueTypeArray))
		writeU32(buf, uint32(GGUFMetadataValueTypeUint32))
		writeU64(buf, uint64(len(t)))
		for _, e := range t {
			writeU32(buf, e)
		}
	case []uint64:
		writeU32(buf, uint32(GGUFMetadataValueTypeArray))
		writeU32(buf, uint32(GGUFMetadataValueTypeUint64))
		writeU64(buf, uint64(len(t)))
		for _, e := range t {
			writeU64(buf, e)
		}
	case []int32:
		writeU32(buf, uint32(GGUFMetadataValueTypeArray))
		writeU32(buf, uint32(GGUFMetadataValueTypeInt32))
		writeU64(buf, uint64(len(t)))
		for _, e := range t {
			writeU32(buf, uint32(e))
		}
	case []int64:
		writeU32(buf, uint32(GGUFMetadataValueTypeArray))
		writeU32(buf, uint32(GGUFMetadataValueTypeInt64))
		writeU64(buf, uint64(len(t)))
		for _, e := range t {
			writeU64(buf, uint64(e))
		}
	case []float32:
		writeU32(buf, uint32(GGUFMetadataValueTypeArray))
		writeU32(buf, uint32(GGUFMetadataValueTypeFloat32))
		writeU64(buf, uint64(len(t)))
		for _, e := range t {
			writeU32(buf, math.Float32bits(e))
		}
	case []float64:
		writeU32(buf, uint32(GGUFMetadataValueTypeArray))
		writeU32(buf, uint32(GGUFMetadataValueTypeFloat64))
		writeU64(buf, uint64(len(t)))
		for _, e := range t {
			writeU64(buf, math.Float64bits(e))
		}
	case []interface{}:
		// A heterogeneous array. Every element has to be the same type,
		// because the GGUF array type is per-array.
		if len(t) == 0 {
			return fmt.Errorf("an empty array has no element type")
		}
		elemType, err := valueTypeOf(t[0])
		if err != nil {
			return err
		}
		for i, e := range t {
			other, err := valueTypeOf(e)
			if err != nil {
				return err
			}
			if other != elemType {
				return fmt.Errorf("the array element %d has type %d, but the first element has type %d", i, other, elemType)
			}
		}
		writeU32(buf, uint32(GGUFMetadataValueTypeArray))
		writeU32(buf, uint32(elemType))
		writeU64(buf, uint64(len(t)))
		for _, e := range t {
			if err := writeRawValue(buf, e); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("unsupported metadata value type %T", v)
	}
	return nil
}

// valueTypeOf returns the GGUF metadata type of a value.
func valueTypeOf(v interface{}) (GGUFMetadataValueType, error) {
	switch v.(type) {
	case uint8:
		return GGUFMetadataValueTypeUint8, nil
	case int8:
		return GGUFMetadataValueTypeInt8, nil
	case uint16:
		return GGUFMetadataValueTypeUint16, nil
	case int16:
		return GGUFMetadataValueTypeInt16, nil
	case uint32:
		return GGUFMetadataValueTypeUint32, nil
	case int32:
		return GGUFMetadataValueTypeInt32, nil
	case uint64:
		return GGUFMetadataValueTypeUint64, nil
	case int64:
		return GGUFMetadataValueTypeInt64, nil
	case float32:
		return GGUFMetadataValueTypeFloat32, nil
	case float64:
		return GGUFMetadataValueTypeFloat64, nil
	case bool:
		return GGUFMetadataValueTypeBool, nil
	case string:
		return GGUFMetadataValueTypeString, nil
	default:
		return 0, fmt.Errorf("unsupported metadata value type %T", v)
	}
}

// writeRawValue writes a value without its type prefix.
func writeRawValue(buf *bytes.Buffer, v interface{}) error {
	switch t := v.(type) {
	case uint8:
		buf.WriteByte(t)
	case int8:
		buf.WriteByte(byte(t))
	case uint16:
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], t)
		buf.Write(b[:])
	case int16:
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], uint16(t))
		buf.Write(b[:])
	case uint32:
		writeU32(buf, t)
	case int32:
		writeU32(buf, uint32(t))
	case uint64:
		writeU64(buf, t)
	case int64:
		writeU64(buf, uint64(t))
	case float32:
		writeU32(buf, math.Float32bits(t))
	case float64:
		writeU64(buf, math.Float64bits(t))
	case bool:
		if t {
			buf.WriteByte(1)
		} else {
			buf.WriteByte(0)
		}
	case string:
		writeString(buf, t)
	default:
		return fmt.Errorf("unsupported metadata value type %T", v)
	}
	return nil
}
