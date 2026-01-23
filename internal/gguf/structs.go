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
	GGMLTypeQ5_0   GGMLType = 4
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
	Offset     uint64 // Offset in the file (absolute or relative to data start?) -> Relative to data start usually
	Data       []byte // Slice of the mmap'd file
}

type GGUFFile struct {
	Header  GGUFHeader
	KV      map[string]interface{}
	Tensors []*TensorInfo
	Data    []byte // The raw mmap'd data
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
