package gguf

import "fmt"

const (
	GGUFMagic   = 0x46554747 // "GGUF"
	GGUFVersion = 3
)

type GGMLType uint32

const (
	GGMLTypeF32    GGMLType = 0
	GGMLTypeF16    GGMLType = 1
	GGMLTypeQ4_0   GGMLType = 2
	GGMLTypeQ4_1   GGMLType = 3
	GGMLTypeQ5_0   GGMLType = 6
	GGMLTypeQ8_0   GGMLType = 8
	GGMLTypeQ2_K   GGMLType = 10
	GGMLTypeQ3_K   GGMLType = 11
	GGMLTypeQ4_K   GGMLType = 12
	GGMLTypeQ5_K   GGMLType = 13
	GGMLTypeQ6_K   GGMLType = 14
	GGMLTypeQ8_K   GGMLType = 15
	GGMLTypeQ4_K_S GGMLType = 99 // Deprecated/Unused
)

type GGUFMetadataValueType uint32

const (
	GGUFMetadataValueTypeUint8   GGUFMetadataValueType = 0
	GGUFMetadataValueTypeInt8    GGUFMetadataValueType = 1
	GGUFMetadataValueTypeUint16  GGUFMetadataValueType = 2
	GGUFMetadataValueTypeInt16   GGUFMetadataValueType = 3
	GGUFMetadataValueTypeUint32  GGUFMetadataValueType = 4
	GGUFMetadataValueTypeInt32   GGUFMetadataValueType = 5
	GGUFMetadataValueTypeFloat32 GGUFMetadataValueType = 6
	GGUFMetadataValueTypeBool    GGUFMetadataValueType = 7
	GGUFMetadataValueTypeString  GGUFMetadataValueType = 8
	GGUFMetadataValueTypeArray   GGUFMetadataValueType = 9
	GGUFMetadataValueTypeUint64  GGUFMetadataValueType = 10
	GGUFMetadataValueTypeInt64   GGUFMetadataValueType = 11
	GGUFMetadataValueTypeFloat64 GGUFMetadataValueType = 12
)

type TensorInfo struct {
	Name       string
	Dimensions []uint64 // ne (number of elements) in each dimension
	Type       GGMLType
	Offset     uint64 // Offset relative to data start
	Data       []byte // Byte slice into the mmap'd file
}

func (t *TensorInfo) SizeBytes() uint64 {
	numElements := uint64(1)
	for _, d := range t.Dimensions {
		numElements *= d
	}

	switch t.Type {
	case GGMLTypeF32:
		return numElements * 4
	case GGMLTypeF16:
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
	default:
		return 0
	}
}

type GGUFFile struct {
	Header     GGUFHeader
	KV         map[string]interface{}
	Tensors    []*TensorInfo
	Data       []byte // The raw mmap'd data
	DataOffset uint64 // Offset where the tensor data starts
}

type GGUFHeader struct {
	Magic       uint32
	Version     uint32
	TensorCount uint64
	KVCount     uint64
}

// Error types
type ErrInvalidMagic struct{ Magic uint32 }

func (e ErrInvalidMagic) Error() string {
	return fmt.Sprintf("invalid GGUF magic: %x", e.Magic)
}

type ErrUnsupportedVersion struct{ Version uint32 }

func (e ErrUnsupportedVersion) Error() string {
	return fmt.Sprintf("unsupported GGUF version: %d", e.Version)
}

func (t GGMLType) String() string {
	switch t {
	case GGMLTypeF32:
		return "F32"
	case GGMLTypeF16:
		return "F16"
	case GGMLTypeQ4_0:
		return "Q4_0"
	case GGMLTypeQ4_1:
		return "Q4_1"
	case GGMLTypeQ5_0:
		return "Q5_0"
	case GGMLTypeQ8_0:
		return "Q8_0"
	case GGMLTypeQ2_K:
		return "Q2_K"
	case GGMLTypeQ3_K:
		return "Q3_K"
	case GGMLTypeQ4_K:
		return "Q4_K"
	case GGMLTypeQ5_K:
		return "Q5_K"
	case GGMLTypeQ6_K:
		return "Q6_K"
	case GGMLTypeQ8_K:
		return "Q8_K"
	case GGMLTypeQ4_K_S:
		return "Q4_K_S"
	default:
		return fmt.Sprintf("UNKNOWN_TYPE_%d", t)
	}
}
