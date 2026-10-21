//go:build linux || darwin

package vector

// mmapstore.go implements P0-05: a memory-mapped vector store.
//
// The flat float32 hot path (P0-01) scores rows through contiguous
// batch kernels; a file the kernel maps into the address space is the
// cheapest such store — the pages never enter the Go heap, they are
// pre-faulted once (MAP_POPULATE + MADV_WILLNEED) so the first query
// pays no soft page-fault storm, and MADV_DONTNEED gives the kernel
// the hint to evict cold ranges, which is what keeps a million-vector
// index inside a 256 MB working set.
//
// Persistence is "zero-copy chunked": rows stream from the source
// index into an mmap-backed file in bounded chunks (no whole-index
// buffer is ever built, so peak RSS stays at a few pages per chunk),
// and loading maps the file and hands out rows that alias the mapped
// bytes directly — the batch kernels score them with zero copies and
// no contiguous mirror allocation (the in-memory path pays a 2x mirror;
// the mmap path does not).

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"syscall"
	"unsafe"
)

var (
	// ErrMMapBadFormat reports a file that is not an LQ vector file.
	ErrMMapBadFormat = errors.New("vector: not an LQ vector file")
	// ErrMMapTooShort reports a file truncated below its header/rows.
	ErrMMapTooShort = errors.New("vector: vector file truncated")
	// ErrMMapClosed reports use of a closed store.
	ErrMMapClosed = errors.New("vector: mmap store closed")
)

// File format magic/header (little endian):
//
//	flat float32: "LQFLT1\x00\x00" | count uint64 | dim uint64 | rows
//	complex SoA : "LQCSOA1\x00"   | count uint64 | dim uint64 | re plane | im plane
const mmapHeaderSize = 24

var mmapMagicFloat32 = [8]byte{'L', 'Q', 'F', 'L', 'T', '1', 0, 0}
var mmapMagicComplexSoA = [8]byte{'L', 'Q', 'C', 'S', 'O', 'A', '1', 0}

// mmapCopyChunk bounds one streaming chunk to 16 MiB of payload for
// the save path (no whole-index buffers; peak RSS = a few pages).
const mmapCopyChunk = 16 << 20

// MMapStore holds one mapped file. The zero value is not usable; use
// OpenMMapStore. The data slice aliases the mapped pages and stays
// valid until Close — which invalidates every row view handed out
// through the regions (and any FlatIndex whose rows alias them).
type MMapStore struct {
	path string
	file *os.File // nil after Close
	data []byte   // aliases the map; nil after Close
	ro   bool
}

// OpenMMapStore maps path read-only, pre-faults the mapping (populate
// + willneed) and returns the store. Callers own the store and must
// Close it exactly once; data rows alias the mapping.
func OpenMMapStore(path string) (*MMapStore, error) {
	return openMMap(path, true)
}

func openMMap(path string, ro bool) (*MMapStore, error) {
	flag := os.O_RDONLY
	if !ro {
		flag = os.O_RDWR
	}
	f, err := os.OpenFile(path, flag, 0)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	size := st.Size()
	if size < mmapHeaderSize {
		f.Close()
		return nil, ErrMMapTooShort
	}
	prot := syscall.PROT_READ
	flags := syscall.MAP_PRIVATE | mmapPopulate
	if !ro {
		prot |= syscall.PROT_WRITE
		flags = syscall.MAP_SHARED
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), prot, flags)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("vector: mmap %s: %w", path, err)
	}
	m := &MMapStore{path: path, file: f, data: data, ro: ro}
	if ro {
		// Pre-fault hint on top of MAP_POPULATE for the platforms
		// where populate does not apply (macOS): read-ahead the whole
		// range so the first queries do not stall on faults.
		_ = m.Prefetch()
	}
	return m, nil
}

// Path returns the backing file name.
func (m *MMapStore) Path() string {
	if m == nil {
		return ""
	}
	return m.path
}

// Data returns the mapped bytes (nil after Close). The slice aliases
// kernel-managed pages: it is not Go-memory, and mutating it through
// a read-only store is a fault.
func (m *MMapStore) Data() []byte {
	if m == nil {
		return nil
	}
	return m.data
}

// Size returns the mapped size in bytes.
func (m *MMapStore) Size() int64 {
	if m == nil || m.data == nil {
		return 0
	}
	return int64(len(m.data))
}

// Close syncs (read-write maps), unmaps and releases the fd. It is
// safe to call twice; rows and regions previously handed out are
// invalid afterwards.
func (m *MMapStore) Close() error {
	if m == nil || m.data == nil {
		return nil
	}
	var err error
	if !m.ro {
		if e := msyncData(m.data); e != nil {
			err = e
		}
	}
	if e := syscall.Munmap(m.data); e != nil && err == nil {
		err = e
	}
	m.data = nil
	if m.file != nil {
		if e := m.file.Close(); e != nil && err == nil {
			err = e
		}
		m.file = nil
	}
	return err
}

// msyncData flushes a read-write map; chunked like the rest of the
// API so very large files are not flushed in one call. The syscall
// shim lives in the per-GOOS mmap_constants files (darwin lacks the
// syscall.Msync wrapper).
func msyncData(data []byte) error {
	const chunk = 1 << 30
	for off := 0; off < len(data); off += chunk {
		hi := off + chunk
		if hi > len(data) {
			hi = len(data)
		}
		if err := msyncSlice(data[off:hi]); err != nil {
			return err
		}
	}
	return nil
}

// Prefetch applies the pre-fault/read-ahead hint (MADV_WILLNEED) to
// the whole mapping (P0-05 "pre-faulting").
func (m *MMapStore) Prefetch() error {
	if m == nil || m.data == nil {
		return ErrMMapClosed
	}
	return madviseRange(m.data, 0, len(m.data), syscall.MADV_WILLNEED)
}

// Evict applies the page-eviction hint (MADV_DONTNEED) to the byte
// range [lo, hi): the kernel may reclaim the clean pages whenever
// memory is needed, which is how a store larger than the working set
// stays resident only where it is used (P0-05).
func (m *MMapStore) Evict(lo, hi int) error {
	if m == nil || m.data == nil {
		return ErrMMapClosed
	}
	if lo < 0 || hi > len(m.data) || lo > hi {
		return fmt.Errorf("vector: evict range [%d,%d) out of bounds", lo, hi)
	}
	return madviseRange(m.data, lo, hi, syscall.MADV_DONTNEED)
}

// madviseRange advises [lo, hi) in ~1 GiB slices (the system call
// takes an int length); the syscall shim is per-GOOS.
func madviseRange(data []byte, lo, hi, advice int) error {
	const chunk = 1 << 30
	page := os.Getpagesize()
	for off := lo; off < hi; off += chunk {
		start := off - off%page // align on page boundaries
		hiChunk := off + chunk
		if hiChunk > hi {
			hiChunk = hi
		}
		if start >= hiChunk {
			continue
		}
		if err := madviseSlice(data[start:hiChunk], advice); err != nil && err != syscall.EINVAL {
			return err
		}
	}
	return nil
}

// Float32Region is a dim-strided float32 view over mapped bytes. Rows
// alias the map; Close invalidates them.
type Float32Region struct {
	plane []float32
	dim   int
}

// Float32RegionAt builds the typed view of a mapped float32 file.
// data must be the full file image (header included); return offset
// errors otherwise.
func Float32RegionAt(data []byte) (Float32Region, error) {
	if len(data) < mmapHeaderSize {
		return Float32Region{}, ErrMMapTooShort
	}
	var magic [8]byte
	copy(magic[:], data[:8])
	if magic != mmapMagicFloat32 {
		return Float32Region{}, ErrMMapBadFormat
	}
	count := int(binary.LittleEndian.Uint64(data[8:16]))
	dim := int(binary.LittleEndian.Uint64(data[16:24]))
	if count < 0 || dim < 0 || mmapHeaderSize+count*dim*4 > len(data) {
		return Float32Region{}, ErrMMapTooShort
	}
	return Float32Region{
		plane: bytesAsFloat32(data[mmapHeaderSize : mmapHeaderSize+count*dim*4]),
		dim:   dim,
	}, nil
}

// Count returns the number of rows.
func (r Float32Region) Count() int {
	if r.dim <= 0 {
		return 0
	}
	return len(r.plane) / r.dim
}

// Dim returns the row stride.
func (r Float32Region) Dim() int {
	return r.dim
}

// Row returns row i aliasing the mapping (len == dim).
func (r Float32Region) Row(i int) []float32 {
	return r.plane[i*r.dim : (i+1)*r.dim]
}

// Rows returns every row aliasing the mapping.
func (r Float32Region) Rows() [][]float32 {
	n := r.Count()
	rows := make([][]float32, n)
	for i := range rows {
		rows[i] = r.Row(i)
	}
	return rows
}

// Plane returns the whole row-major plane (stride dim), the layout the
// batch kernels score.
func (r Float32Region) Plane() []float32 {
	return r.plane
}

// bytesAsFloat32 reinterprets a 4-byte aligned byte slice as float32.
func bytesAsFloat32(data []byte) []float32 {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), len(data)/4)
}

// bytesAsFloat64 reinterprets an 8-byte aligned byte slice as float64.
func bytesAsFloat64(data []byte) []float64 {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Slice((*float64)(unsafe.Pointer(&data[0])), len(data)/8)
}

// writeMMapHeader writes magic + count + dim and truncates the file to
// header + payload size. It is the file-creation entry point of every
// save path.
func writeMMapHeader(path string, magic [8]byte, count, dim uint64) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	var head [mmapHeaderSize]byte
	copy(head[:8], magic[:])
	binary.LittleEndian.PutUint64(head[8:16], count)
	binary.LittleEndian.PutUint64(head[16:24], dim)
	if _, err := f.Write(head[:]); err != nil {
		return err
	}
	payload := int64(count) * int64(dim)
	switch magic {
	case mmapMagicFloat32:
		payload *= 4
	case mmapMagicComplexSoA:
		payload *= 16 // two float64 planes
	}
	if err := f.Truncate(mmapHeaderSize + payload); err != nil {
		return err
	}
	return nil
}

// chunkRows bounds the rows one save chunk covers: at most 16 MiB of
// payload, and at least one row whatever dim is.
func chunkRows(dim int) int {
	rows := mmapCopyChunk / (dim * 4)
	if rows < 1 {
		rows = 1
	}
	return rows
}

// SaveFlatIndexFloat32 persists idx.Float32 row-major (little endian)
// to path through an mmap'd file in bounded chunks: no whole-index
// buffer is built, so peak RSS stays at a few pages of chunk scratch.
// The stored values are bit-identical to the in-memory rows.
func SaveFlatIndexFloat32(path string, idx *FlatIndex) error {
	if idx == nil {
		return fmt.Errorf("vector: nil index")
	}
	return saveFlatIndexFloat32(path, idx)
}

// saveFlatIndexFloat32 is the SaveFlatIndexFloat32 body (split out for
// readability).
func saveFlatIndexFloat32(path string, idx *FlatIndex) error {
	dim := idx.Dimension
	n := len(idx.Float32)
	if dim <= 0 && n > 0 {
		return fmt.Errorf("vector: cannot save %d rows with dim 0", n)
	}
	if err := writeMMapHeader(path, mmapMagicFloat32, uint64(n), uint64(dim)); err != nil {
		return err
	}
	if n == 0 {
		return nil
	}
	// Map the freshly created file and stream the rows into the
	// mapping in chunkRows blocks, syncing once at the end. The plane
	// aliases the whole map; the first mmapHeaderSize/4 floats are the
	// header (24 = 6 floats, exactly convertible), so rows start after
	// that prefix.
	f, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	total := int64(n) * int64(dim) * 4
	data, err := syscall.Mmap(int(f.Fd()), 0, int(mmapHeaderSize+total), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return fmt.Errorf("vector: mmap %s for save: %w", path, err)
	}
	plane := bytesAsFloat32(data)
	skip := mmapHeaderSize / 4
	rows := chunkRows(dim)
	for start := 0; start < n; start += rows {
		end := start + rows
		if end > n {
			end = n
		}
		for i := start; i < end; i++ {
			copy(plane[skip+i*dim:skip+(i+1)*dim], idx.Float32[i][:dim])
		}
	}
	if err := msyncData(data); err != nil {
		syscall.Munmap(data)
		return err
	}
	if err := syscall.Munmap(data); err != nil {
		return err
	}
	return nil
}

// LoadFlatIndexFloat32 maps the flat float32 file at path and returns
// an index whose rows alias the mapping: scoring it load zero pages
// into the Go heap, and PrepareFloat32 finds the plane already
// contiguous, so no mirror copy is made either. Callers must keep the
// returned store alive (and Close it) for as long as they search; a
// Close invalidates the index.
func LoadFlatIndexFloat32(path string) (*FlatIndex, *MMapStore, error) {
	store, err := OpenMMapStore(path)
	if err != nil {
		return nil, nil, err
	}
	region, err := Float32RegionAt(store.data)
	if err != nil {
		_ = store.Close()
		return nil, nil, err
	}
	idx := NewFlatIndex(region.dim, TypeFloat32)
	rows := region.Rows()
	idx.Float32 = rows
	idx.f32mirror = region.plane // zero-copy: rows are contiguous already
	idx.f32Uniform = region.dim > 0
	return idx, store, nil
}

// ComplexSoARegion is the typed view over a mapped complex SoA file.
type ComplexSoARegion struct {
	re  []float64
	im  []float64
	n   int
	dim int
}

// ComplexSoARegionAt builds the typed view of a mapped SoA file.
func ComplexSoARegionAt(data []byte) (ComplexSoARegion, error) {
	if len(data) < mmapHeaderSize {
		return ComplexSoARegion{}, ErrMMapTooShort
	}
	var magic [8]byte
	copy(magic[:], data[:8])
	if magic != mmapMagicComplexSoA {
		return ComplexSoARegion{}, ErrMMapBadFormat
	}
	count := int(binary.LittleEndian.Uint64(data[8:16]))
	dim := int(binary.LittleEndian.Uint64(data[16:24]))
	if count < 0 || dim < 0 {
		return ComplexSoARegion{}, ErrMMapBadFormat
	}
	planeBytes := count * dim * 8
	if mmapHeaderSize+2*planeBytes > len(data) {
		return ComplexSoARegion{}, ErrMMapTooShort
	}
	return ComplexSoARegion{
		re:  bytesAsFloat64(data[mmapHeaderSize : mmapHeaderSize+planeBytes]),
		im:  bytesAsFloat64(data[mmapHeaderSize+planeBytes : mmapHeaderSize+2*planeBytes]),
		n:   count,
		dim: dim,
	}, nil
}

// SoA returns the split planes as a ComplexSoA aliasing the mapping.
func (r ComplexSoARegion) SoA() *ComplexSoA {
	return &ComplexSoA{Count: r.n, Dim: r.dim, Re: r.re, Im: r.im}
}

// SaveComplexSoA persists the split real/imag planes of idx (the SoA
// representation built by PrepareComplex) to path in bounded chunks,
// mirroring the float32 save path. The zero-copy payoff is on the load
// side: LoadFlatIndexComplexSoA scores from the mapped pages directly.
func SaveComplexSoA(path string, idx *FlatIndex) error {
	if idx == nil {
		return fmt.Errorf("vector: nil index")
	}
	soa := idx.PrepareComplex()
	if soa == nil {
		return fmt.Errorf("vector: no SoA representation (ragged or empty rows)")
	}
	if err := writeMMapHeader(path, mmapMagicComplexSoA, uint64(soa.Count), uint64(soa.Dim)); err != nil {
		return err
	}
	if soa.Count == 0 {
		return nil
	}
	f, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	total := int64(soa.Count) * int64(soa.Dim) * 8
	// One contiguous mapping over header + both planes; the plane
	// prefix is the header in float64 units (24 = 3 doubles), so
	// rows start at skip, and im right after re.
	data, err := syscall.Mmap(int(f.Fd()), 0, int(mmapHeaderSize+2*total), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return fmt.Errorf("vector: mmap %s for save: %w", path, err)
	}
	plane := bytesAsFloat64(data)
	skip := mmapHeaderSize / 8
	nelem := soa.Count * soa.Dim
	rows := mmapCopyChunk / (soa.Dim * 8)
	if rows < 1 {
		rows = 1
	}
	for start := 0; start < soa.Count; start += rows {
		end := start + rows
		if end > soa.Count {
			end = soa.Count
		}
		copy(plane[skip+start*soa.Dim:skip+end*soa.Dim], soa.Re[start*soa.Dim:end*soa.Dim])
		copy(plane[skip+nelem+start*soa.Dim:skip+nelem+end*soa.Dim], soa.Im[start*soa.Dim:end*soa.Dim])
	}
	if err := msyncData(data); err != nil {
		syscall.Munmap(data)
		return err
	}
	if err := syscall.Munmap(data); err != nil {
		return err
	}
	return nil
}

// LoadFlatIndexComplexSoA maps the SoA file at path and returns an
// index whose SoA planes alias the mapping (zero-copy scoring path of
// P0-02/P0-05). The interleaved view is not materialised; norms are
// computed once for cosine. Callers must keep the store alive.
func LoadFlatIndexComplexSoA(path string) (*FlatIndex, *MMapStore, error) {
	store, err := OpenMMapStore(path)
	if err != nil {
		return nil, nil, err
	}
	region, err := ComplexSoARegionAt(store.data)
	if err != nil {
		_ = store.Close()
		return nil, nil, err
	}
	soa := region.SoA()
	idx := NewFlatIndex(region.dim, TypeComplex128)
	idx.complexSoA = soa
	idx.complexCount = region.n
	idx.complexNorms = make([]float64, region.n)
	for i := 0; i < region.n; i++ {
		re, im := soa.Row(i)
		idx.complexNorms[i] = dotComplex128Scalars(re, im, re, im)
	}
	return idx, store, nil
}

// OpenCountFor reports how many of the process' open file descriptors
// point at path (resolved symlinks, linux /proc/self/fd). It returns
// -1 where the probe is unavailable (/proc absent, or target reads
// failed), so leak tests must skip on -1 rather than fail.
func OpenCountFor(path string) int {
	entries, err := os.ReadDir("/proc/self/fd")
	if err != nil {
		return -1
	}
	target, err := filepath.EvalSymlinks(path)
	if err != nil {
		target = path
	}
	count := 0
	for _, e := range entries {
		link, err := os.Readlink(filepath.Join("/proc/self/fd", e.Name()))
		if err != nil {
			continue
		}
		if link == path || (target != "" && link == target) {
			count++
		}
	}
	return count
}
