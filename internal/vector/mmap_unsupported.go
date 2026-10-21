//go:build !linux && !darwin

package vector

// mmap_unsupported.go keeps the public mmap API type-correct on
// platforms without the POSIX mmap paths this store relies on (the
// shipped kernels and stores target linux; darwin shares the code).
// Every entry point reports an error so callers degrade to the
// in-memory stores (the tests skip on unsupported platforms).

import (
	"errors"
)

var (
	ErrMMapBadFormat = errors.New("vector: not an LQ vector file")
	ErrMMapTooShort  = errors.New("vector: vector file truncated")
	ErrMMapClosed    = errors.New("vector: mmap store closed")
	// ErrMMapUnsupported marks every failed entry point on platforms
	// without the mmap implementation.
	ErrMMapUnsupported = errors.New("vector: mmap store not supported on this platform")
)

// MMapStore is the inert counterpart of the POSIX store; every method
// returns ErrMMapUnsupported or harmless zero values.
type MMapStore struct{}

// OpenMMapStore always fails on unsupported platforms.
func OpenMMapStore(path string) (*MMapStore, error) {
	return nil, ErrMMapUnsupported
}

// Path is always empty.
func (m *MMapStore) Path() string { return "" }

// Data is always nil.
func (m *MMapStore) Data() []byte { return nil }

// Size is always zero.
func (m *MMapStore) Size() int64 { return 0 }

// Close is a no-op returning an error.
func (m *MMapStore) Close() error { return ErrMMapUnsupported }

// Prefetch always fails.
func (m *MMapStore) Prefetch() error { return ErrMMapUnsupported }

// Evict always fails.
func (m *MMapStore) Evict(lo, hi int) error { return ErrMMapUnsupported }

// Float32Region is an empty typed view (Count 0).
type Float32Region struct{}

// Count is zero.
func (r Float32Region) Count() int { return 0 }

// Dim is zero.
func (r Float32Region) Dim() int { return 0 }

// SaveFlatIndexFloat32 always fails on unsupported platforms.
func SaveFlatIndexFloat32(path string, idx *FlatIndex) error {
	return ErrMMapUnsupported
}

// LoadFlatIndexFloat32 always fails on unsupported platforms.
func LoadFlatIndexFloat32(path string) (*FlatIndex, *MMapStore, error) {
	return nil, nil, ErrMMapUnsupported
}

// ComplexSoARegion is an empty typed view.
type ComplexSoARegion struct{}

// SaveComplexSoA always fails on unsupported platforms.
func SaveComplexSoA(path string, idx *FlatIndex) error {
	return ErrMMapUnsupported
}

// LoadFlatIndexComplexSoA always fails on unsupported platforms.
func LoadFlatIndexComplexSoA(path string) (*FlatIndex, *MMapStore, error) {
	return nil, nil, ErrMMapUnsupported
}

// OpenCountFor reports fd counts only where /proc exists; unsupported
// platforms return -1 (tests skip on it).
func OpenCountFor(path string) int {
	return -1
}
