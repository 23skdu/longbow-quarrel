//go:build (!cuda && !metal && !tpu) || !amd64 || !cgo || (!linux && !darwin)

package device

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync/atomic"
	"unsafe"

	"github.com/23skdu/longbow-quarrel/internal/simd"
)

var ErrOOM = errors.New("OOM: memory limit exceeded")

type Context struct {
	device     int
	memUsed    int64
	numThreads int

	// TurboQuant Global Matrices
	TQRotation *Tensor
	TQQJL      *Tensor

	// Performance Counters (Hotpath)
	ArrowBytesProcessed atomic.Int64
}

func AllocatedBytes() int64 {
	return 0
}

func NewContext() *Context {
	return &Context{
		device:     -1,
		memUsed:    0,
		numThreads: runtime.NumCPU(),
	}
}

func (c *Context) DeviceID() int {
	return c.device
}

func (c *Context) Synchronize() {
	// No-op for CPU
}

func (c *Context) Free() {
	c.memUsed = 0
}

func (c *Context) LoadBuffer(t *Tensor, data []byte) {
	if t.rawData != nil && len(data) <= len(t.rawData) {
		copy(t.rawData, data)
	}
}

func checkMem(bytes int) error {
	maxMB := atomic.LoadInt64(&maxMemoryMB)
	if maxMB == 0 {
		return nil
	}
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	if m.Alloc > uint64(maxMB)*1024*1024 { // #nosec G115 -- maxMB is always positive
		runtime.GC()
		runtime.ReadMemStats(&m)
		if m.Alloc > uint64(maxMB)*1024*1024 { // #nosec G115 -- maxMB is always positive
			return ErrOOM
		}
	}
	if m.Alloc+uint64(bytes) > uint64(maxMB)*1024*1024 { // #nosec G115 -- maxMB is always positive
		runtime.GC()
		runtime.ReadMemStats(&m)
		if m.Alloc+uint64(bytes) > uint64(maxMB)*1024*1024 { // #nosec G115 -- maxMB is always positive
			return ErrOOM
		}
	}
	return nil
}

func (c *Context) NewTensor(rows, cols int) *Tensor {
	if err := checkMem(rows * cols * 4); err != nil {
		panic(err)
	}
	return &Tensor{
		ctx:      c,
		data:     make([]float32, rows*cols),
		dims:     []int{rows, cols},
		strides:  []int{cols, 1},
		dataType: DataTypeF32,
	}
}

func (c *Context) NewTensorFP32(rows, cols int) *Tensor {
	return c.NewTensor(rows, cols)
}

func (c *Context) NewTensorFP32Pooled(rows, cols int) *Tensor {
	return c.NewTensor(rows, cols)
}

func (c *Context) NewTensorPooled(rows, cols int) *Tensor {
	return c.NewTensor(rows, cols)
}
func (c *Context) NewTurboTensor(rows, cols int, dt DataType, blockSize, qjlRows int) *Tensor {
	if blockSize > 0 {
		numElements := rows * cols
		numBlocks := numElements / blockSize
		if numElements%blockSize != 0 {
			numBlocks++
		}
		bytesPerBlock := blockSize + qjlRows + 8
		if err := checkMem(numBlocks * bytesPerBlock); err != nil {
			panic(err)
		}
	}
	t := &Tensor{
		ctx:       c,
		dims:      []int{rows, cols},
		strides:   []int{cols, 1},
		dataType:  dt,
		blockSize: blockSize,
		qjlRows:   qjlRows,
	}
	numElements := rows * cols
	if blockSize > 0 {
		numBlocks := numElements / blockSize
		if numElements%blockSize != 0 {
			numBlocks++
		}
		bytesPerBlock := blockSize + qjlRows + 8 // Polar + QJL + 2 Scales
		t.rawData = make([]byte, numBlocks*bytesPerBlock)
	}
	return t
}

func (c *Context) NewTensorWithType(rows, cols int, dt DataType) *Tensor {
	if dt == DataTypeTQ1_0 || dt == DataTypeTQ2_0 {
		// Use standard TurboQuant defaults (128/64) for general weighting
		return c.NewTurboTensor(rows, cols, dt, 128, 64)
	}
	t := &Tensor{
		ctx:      c,
		dims:     []int{rows, cols},
		strides:  []int{cols, 1},
		dataType: dt,
	}
	numElements := rows * cols
	switch dt {
	case DataTypeINT8:
		t.rawData = make([]byte, numElements)
	case DataTypeF32:
		t.data = make([]float32, numElements)
	default:
		t.data = make([]float32, numElements)
	}
	return t
}

type Tensor struct {
	ctx       *Context
	data      []float32
	rawData   []byte // Used for quantized formats (TQ1, TQ2, Q4K, etc.)
	dims      []int
	strides   []int
	name      string
	dataType  DataType
	blockSize int // For TurboQuant
	qjlRows   int // For TurboQuant
}

func NewTensor(name string, data []float32) *Tensor {
	dims := []int{len(data)}
	strides := []int{1}
	return &Tensor{
		data:     data,
		dims:     dims,
		strides:  strides,
		name:     name,
		dataType: DataTypeF32,
	}
}

func (t *Tensor) RawData() []byte {
	if t.data != nil {
		return unsafe.Slice((*byte)(unsafe.Pointer(&t.data[0])), len(t.data)*4) // #nosec G103
	}
	return t.rawData
}

func (t *Tensor) Name() string {
	return t.name
}

func (t *Tensor) Data() []float32 {
	return t.data
}

func (t *Tensor) SizeBytes() int {
	if t.dataType == DataTypeF32 {
		return len(t.data) * 4
	}
	return len(t.rawData)
}

func (t *Tensor) LoadFromF32(data []float32) error {
	return t.LoadFrom(data)
}

func (t *Tensor) ToHostF32() []float32 {
	return t.data
}

func (t *Tensor) CopyToF16() *Tensor {
	// For CPU, we keep it as is for now or could implement half conversion
	return t
}

func (t *Tensor) Free() {
	t.data = nil
	t.rawData = nil
}

func (t *Tensor) ZeroInit() {
	for i := range t.data {
		t.data[i] = 0
	}
	for i := range t.rawData {
		t.rawData[i] = 0
	}
}

func (t *Tensor) DataType() DataType { return t.dataType }

func (t *Tensor) IsDevice() bool { return false }

func (t *Tensor) Rows() int {
	if len(t.dims) < 1 {
		return 0
	}
	return t.dims[0]
}

func (t *Tensor) Cols() int {
	if len(t.dims) < 2 {
		return 1
	}
	return t.dims[1]
}

func (t *Tensor) FetchKV(v *Tensor, kCache, vCache *Tensor, pos, heads, headDim, windowSize int) {
	// Standard contiguous cache layout: [windowSize, heads, headDim]
	off := (pos % windowSize) * heads * headDim
	size := heads * headDim

	if kCache.dataType == DataTypeF32 {
		// FP32 Fetch Path
		if t.data != nil && kCache.data != nil && len(kCache.data) >= off+size {
			copy(t.data[:size], kCache.data[off:off+size])
			copy(v.data[:size], vCache.data[off:off+size])
		} else if t.data != nil {
			// Handle rawData backends (e.g. pooled)
			if kCache.rawData != nil && len(kCache.rawData) >= (off+size)*4 {
				kSrc := kCache.rawData[off*4:]
				vSrc := vCache.rawData[off*4:]
				for i := 0; i < size; i++ {
					t.data[i] = getFloat32(kSrc[i*4:])
					v.data[i] = getFloat32(vSrc[i*4:])
				}
			}
		}
		return
	}

	if kCache.dataType == DataTypeTQ1_0 || kCache.dataType == DataTypeTQ2_0 {
		// TurboQuant Fetch Path (Dequantization)
		qjlRows := kCache.qjlRows
		blockSize := kCache.blockSize
		if blockSize == 0 {
			blockSize = headDim // Fallback
		}
		if qjlRows == 0 {
			qjlRows = 64 // Fallback
		}
		bytesPerBlock := blockSize + qjlRows + 8
		cacheOff := (pos % windowSize) * heads * bytesPerBlock

		rot := t.ctx.TQRotation

		for h := 0; h < heads; h++ {
			blockCacheStart := cacheOff + h*bytesPerBlock

			kSrc := kCache.rawData[blockCacheStart:]
			kDest := t.data[h*headDim : (h+1)*headDim]
			dequantizeBlock(t.ctx, kSrc, kDest, blockSize, qjlRows, rot)

			vSrc := vCache.rawData[blockCacheStart:]
			vDest := v.data[h*headDim : (h+1)*headDim]
			dequantizeBlock(t.ctx, vSrc, vDest, blockSize, qjlRows, rot)
		}
	}
}

func dequantizeBlock(c *Context, src []byte, dst []float32, blockSize, qjlRows int, rotationMatrix *Tensor) {
	q := make([]int8, blockSize)
	for i := 0; i < blockSize; i++ {
		q[i] = int8(src[i]) // #nosec G115
	}
	qj := make([]int8, qjlRows)
	for i := 0; i < qjlRows; i++ {
		qj[i] = int8(src[blockSize+i]) // #nosec G115
	}
	s := getFloat32(src[blockSize+qjlRows : blockSize+qjlRows+4])
	sj := getFloat32(src[blockSize+qjlRows+4 : blockSize+qjlRows+8])

	rotatedRes := make([]float32, blockSize)
	for i := 0; i < blockSize; i++ {
		rotatedRes[i] = float32(q[i]) * s
	}

	// Apply Inverse Rotation to get original space
	if rotationMatrix != nil {
		for i := 0; i < blockSize; i++ {
			var sum float32
			for j := 0; j < blockSize; j++ {
				sum += rotationMatrix.data[j*blockSize+i] * rotatedRes[j]
			}
			dst[i] = sum
		}
	} else {
		copy(dst, rotatedRes)
	}

	// 2. Add QJL Residual in Original Space
	if sj > 0 && c.TQQJL != nil {
		for i := 0; i < qjlRows; i++ {
			scale_i := float32(int8(qj[i])) * sj
			for j := 0; j < blockSize; j++ {
				dst[j] += scale_i * c.TQQJL.data[i*blockSize+j]
			}
		}
	}
}

func (t *Tensor) ToHost() []float32 {
	return t.data
}

func (t *Tensor) ToHostFP16() []uint16 {
	res := make([]uint16, len(t.data))
	for i, v := range t.data {
		res[i] = Float32ToFloat16(v)
	}
	return res
}

func (t *Tensor) LoadFrom(data interface{}) error {
	switch d := data.(type) {
	case []float32:
		if len(d) != t.NumElements() {
			return fmt.Errorf("LoadFrom: size mismatch: %d != %d", len(d), t.NumElements())
		}
		copy(t.data, d)
		return nil
	case []byte:
		return t.LoadFromRaw(d)
	default:
		return fmt.Errorf("LoadFrom: unsupported data type: %T", data)
	}
}

// LoadFromRaw copies raw bytes to the tensor (for F32 currently on CPU)
func (t *Tensor) LoadFromRaw(data []byte) error {
	if len(data) > t.SizeBytes() {
		return fmt.Errorf("LoadFromRaw: data size %d exceeds tensor size %d", len(data), t.SizeBytes())
	}
	if t.dataType == DataTypeF32 {
		// Copy bytes to float32 slice
		ptr := unsafe.Pointer(&t.data[0])                      // #nosec G103 -- intentional unsafe for zero-copy
		byteSlice := unsafe.Slice((*byte)(ptr), len(t.data)*4) // #nosec G103 -- intentional unsafe for zero-copy
		copy(byteSlice, data)
	} else if t.rawData != nil {
		copy(t.rawData, data)
	}
	return nil
}

func (t *Tensor) StoreKV(v *Tensor, kCache, vCache *Tensor, pos, heads, headDim, windowSize int) {
	// Standard contiguous cache layout: [windowSize, heads, headDim]
	off := (pos % windowSize) * heads * headDim
	size := heads * headDim

	if kCache.dataType == DataTypeF32 {
		// FP32 Storage Path
		if t.data != nil && kCache.data != nil && len(kCache.data) >= off+size {
			copy(kCache.data[off:off+size], t.data[:size])
			copy(vCache.data[off:off+size], v.data[:size])
		} else if t.data != nil {
			// Handle rawData backends (e.g. pooled)
			if kCache.rawData != nil && len(kCache.rawData) >= (off+size)*4 {
				kDst := kCache.rawData[off*4:]
				vDst := vCache.rawData[off*4:]
				for i, val := range t.data[:size] {
					setFloat32(kDst[i*4:], val)
				}
				for i, val := range v.data[:size] {
					setFloat32(vDst[i*4:], val)
				}
			}
		}
		return
	}

	if kCache.dataType == DataTypeTQ1_0 || kCache.dataType == DataTypeTQ2_0 {
		bits := 2 // At least 2 bits (sign + 1 bit magnitude) for reasonable precision
		if kCache.dataType == DataTypeTQ2_0 {
			bits = 4
		}

		qjlRows := kCache.qjlRows
		blockSize := kCache.blockSize
		if blockSize == 0 {
			blockSize = headDim
		}
		if qjlRows == 0 {
			qjlRows = 64
		}
		bytesPerBlock := blockSize + qjlRows + 8

		// Verify if rawData was allocated with this blockSize
		// If not, we might need a separate field in Tensor for blockSize
		// For now, assume it matches.
		cacheOff := (pos % windowSize) * heads * bytesPerBlock

		for h := 0; h < heads; h++ {
			headStart := h * headDim
			headData := t.data[headStart : headStart+headDim]
			blockCacheStart := cacheOff + h*bytesPerBlock

			rot := t.ctx.TQRotation
			qjl := t.ctx.TQQJL
			if rot == nil || qjl == nil {
				continue
			}

			// Encode K
			q, s, res := simd.PolarQuantSIMD(headData, rot.data, blockSize, bits)
			qj, sj := simd.QJLTransformSIMD(res, qjl.data, qjlRows, blockSize)

			dst := kCache.rawData[blockCacheStart:]
			for i, val := range q {
				dst[i] = byte(val) // #nosec G115 -- int8 to byte for quantized data
			}
			for i, val := range qj {
				dst[blockSize+i] = byte(val) // #nosec G115 -- int8 to byte for quantized data
			}
			setFloat32(dst[blockSize+qjlRows:blockSize+qjlRows+4], s)
			setFloat32(dst[blockSize+qjlRows+4:blockSize+qjlRows+8], sj)

			// Encode V
			vHeadData := v.data[headStart : headStart+headDim]
			qv, sv, resv := simd.PolarQuantSIMD(vHeadData, rot.data, blockSize, bits)
			qjv, sjv := simd.QJLTransformSIMD(resv, qjl.data, qjlRows, blockSize)

			vdst := vCache.rawData[blockCacheStart:]
			for i, val := range qv {
				vdst[i] = byte(val) // #nosec G115 -- int8 to byte for quantized data
			}
			for i, val := range qjv {
				vdst[blockSize+i] = byte(val) // #nosec G115 -- int8 to byte for quantized data
			}
			setFloat32(vdst[blockSize+qjlRows:blockSize+qjlRows+4], sv)
			setFloat32(vdst[blockSize+qjlRows+4:blockSize+qjlRows+8], sjv)
		}
	}
}

func (c *Context) TurboQuantEncode(input *Tensor, rotationMatrix *Tensor, qjlMatrix *Tensor, output *Tensor, scaleOut *Tensor, qjlScaleOut *Tensor, blockSize, qjlRows, bits int) {
	if output.blockSize > 0 {
		blockSize = output.blockSize
	}
	if output.qjlRows > 0 {
		qjlRows = output.qjlRows
	}

	numElements := input.Rows() * input.Cols()
	numBlocks := numElements / blockSize

	for b := 0; b < numBlocks; b++ {
		off := b * blockSize
		in := input.data[off : off+blockSize]

		q, s, res := simd.PolarQuantSIMD(in, rotationMatrix.data, blockSize, bits)
		qj, sj := simd.QJLTransformSIMD(res, qjlMatrix.data, qjlRows, blockSize)

		if output.rawData != nil {
			bytesPerBlock := blockSize + qjlRows + 8
			dst := output.rawData[b*bytesPerBlock:]
			for i, v := range q {
				dst[i] = byte(v) // #nosec G115 -- int8 to byte for quantized data
			}
			for i, v := range qj {
				dst[blockSize+i] = byte(v) // #nosec G115 -- int8 to byte for quantized data
			}
			setFloat32(dst[blockSize+qjlRows:blockSize+qjlRows+4], s)
			setFloat32(dst[blockSize+qjlRows+4:blockSize+qjlRows+8], sj)
		} else {
			copy(output.data[off:off+blockSize], qInt8ToF32(q))
		}

		if scaleOut != nil && len(scaleOut.data) > b {
			scaleOut.data[b] = s
		}
		if qjlScaleOut != nil && len(qjlScaleOut.data) > b {
			qjlScaleOut.data[b] = sj
		}
	}
}

func (c *Context) TurboQuantDecode(input *Tensor, rotationMatrix *Tensor, qjlMatrix *Tensor, output *Tensor, scaleIn *Tensor, blockSize, qjlRows int) {
	if input.blockSize > 0 {
		blockSize = input.blockSize
	}
	if input.qjlRows > 0 {
		qjlRows = input.qjlRows
	}

	numElements := output.Rows() * output.Cols()
	numBlocks := numElements / blockSize

	for b := 0; b < numBlocks; b++ {
		var q []int8
		var qj []int8
		var s float32
		var sj float32

		if input.rawData != nil {
			bytesPerBlock := blockSize + qjlRows + 8
			src := input.rawData[b*bytesPerBlock:]
			q = make([]int8, blockSize)
			for i := 0; i < blockSize; i++ {
				q[i] = int8(src[i]) // #nosec G115 -- byte to int8 for quantized data
			}
			qj = make([]int8, qjlRows)
			for i := 0; i < qjlRows; i++ {
				qj[i] = int8(src[blockSize+i]) // #nosec G115 -- byte to int8 for quantized data
			}
			s = getFloat32(src[blockSize+qjlRows : blockSize+qjlRows+4])
			sj = getFloat32(src[blockSize+qjlRows+4 : blockSize+qjlRows+8])
		} else {
			continue
		}

		// Proper Decoder:
		// 1. Reconstruct rotated part (Polar)
		rotatedRes := make([]float32, blockSize)
		for i := 0; i < blockSize; i++ {
			rotatedRes[i] = float32(q[i]) * s
		}

		// 2. Rotate back to original space: out = R^T * rotatedRes
		out := output.data[b*blockSize : (b+1)*blockSize]
		for i := 0; i < blockSize; i++ {
			var sum float32
			for j := 0; j < blockSize; j++ {
				sum += rotationMatrix.data[j*blockSize+i] * rotatedRes[j]
			}
			out[i] = sum
		}

		// 3. Add QJL contribution (Residual) already in original space
		if sj > 0 && qjlMatrix != nil {
			// Random sign matrix reconstruction factor
			// Residual ≈ (sj / sqrt(blockSize)) * (1/rows) * SignMatrix^T * qj
			// But since sj was computed as RMS(Projected), it already has sqrt(blockSize) bias.
			normFactor := sj / (float32(qjlRows) * float32(math.Sqrt(float64(blockSize))))
			for i := 0; i < qjlRows; i++ {
				scale_i := float32(int8(qj[i])) * normFactor
				for j := 0; j < blockSize; j++ {
					out[j] += scale_i * qjlMatrix.data[i*blockSize+j]
				}
			}
		}
	}
}

func qInt8ToF32(in []int8) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

func (t *Tensor) BufferID() uintptr {
	return 0
}

func (t *Tensor) NumElements() int {
	n := 1
	for _, d := range t.dims {
		n *= d
	}
	return n
}

var cpuAllocatedBytes int64

func CPUAllocatedBytes() int64 {
	return atomic.LoadInt64(&cpuAllocatedBytes)
}

func (c *Context) SetNumThreads(n int) {
	c.numThreads = n
}

func (c *Context) NumThreads() int {
	return c.numThreads
}

// AttentionPagedBatch performs paged attention across a batch of sequences on the CPU.
// q: [batchSize, heads, headDim]
// kCache, vCache: [totalBlocks, blockSize, heads, headDim] (block-paged pool)
// blockTables: [batchSize, maxBlocksPerSeq] (int32 physical block IDs)
func (c *Context) AttentionPagedBatch(q, kCache, vCache, output, tokenPositions, blockTables *Tensor, maxBlocksPerSeq, heads, kvHeads, headDim, blockSize int, tokenToSeq *Tensor, batchSize int) {
	// Reference multi-threaded implementation for CPU
	// This is a naive implementation for numerical verification.

	// Assuming F32 for CPU reference
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	for b := 0; b < batchSize; b++ {
		// Get current token position and block assignments
		pos := int(getFloat32(tokenPositions.rawData[b*4:]))

		for h := 0; h < heads; h++ {
			qOff := (b*heads + h) * headDim
			qHead := q.data[qOff : qOff+headDim]

			scores := make([]float32, pos+1)

			// Compute Attention Scores
			for p := 0; p <= pos; p++ {
				logicalBlockIdx := p / blockSize
				blockOffset := p % blockSize

				// Fetch physical block ID
				pBlockID := int(getFloat32(blockTables.rawData[(b*maxBlocksPerSeq+logicalBlockIdx)*4:]))

				// Map to physical memory in pool
				// Pooling layout: [blockIdx][tokenInBlock][head][dim]
				kOff := ((pBlockID*blockSize+blockOffset)*kvHeads + (h % kvHeads)) * headDim
				kHead := kCache.data[kOff : kOff+headDim]

				var dot float32
				for i := 0; i < headDim; i++ {
					dot += qHead[i] * kHead[i]
				}
				scores[p] = dot * scale
			}

			// Softmax
			simd.SoftmaxAVX2(scores)

			// Weighted Sum
			outOff := (b*heads + h) * headDim
			outHead := output.data[outOff : outOff+headDim]
			for i := range outHead {
				outHead[i] = 0
			}

			for p := 0; p <= pos; p++ {
				logicalBlockIdx := p / blockSize
				blockOffset := p % blockSize
				pBlockID := int(getFloat32(blockTables.rawData[(b*maxBlocksPerSeq+logicalBlockIdx)*4:]))

				vOff := ((pBlockID*blockSize+blockOffset)*kvHeads + (h % kvHeads)) * headDim
				vHead := vCache.data[vOff : vOff+headDim]

				s := scores[p]
				for i := 0; i < headDim; i++ {
					outHead[i] += s * vHead[i]
				}
			}
		}
	}
}

// StoreKVPagedBatch stores K and V projections into their respective physical blocks in the CPU cache pool.
func (c *Context) StoreKVPagedBatch(k, v, kCache, vCache, physicalPositions *Tensor, kvDim, batchSize int) {
	for b := 0; b < batchSize; b++ {
		// physicalPosition is absolute token index in the block pool: blockID * blockSize + offset
		pPos := int(getFloat32(physicalPositions.rawData[b*4:]))

		offSrc := b * kvDim
		offDst := pPos * kvDim

		copy(kCache.data[offDst:offDst+kvDim], k.data[offSrc:offSrc+kvDim])
		copy(vCache.data[offDst:offDst+kvDim], v.data[offSrc:offSrc+kvDim])
	}
}

// StoreKVQuantized stores K and V projections into quantized KV cache (FP8 or INT8/Q8_0).
func (t *Tensor) StoreKVQuantized(v *Tensor, kCache, vCache *Tensor, pos, heads, headDim, windowSize int) {
	if kCache == nil || vCache == nil || t == nil || v == nil || windowSize <= 0 {
		return
	}
	off := (pos % windowSize) * heads * headDim
	size := heads * headDim

	tData := t.ToHostF32()
	vData := v.ToHostF32()
	if len(tData) < size || len(vData) < size {
		return
	}

	if kCache.dataType == DataTypeFP8 {
		if kCache.rawData != nil && len(kCache.rawData) >= off+size &&
			vCache.rawData != nil && len(vCache.rawData) >= off+size {
			for i := 0; i < size; i++ {
				kCache.rawData[off+i] = Float32ToFP8E4M3(tData[i])
				vCache.rawData[off+i] = Float32ToFP8E4M3(vData[i])
			}
			return
		}
	}

	if kCache.dataType == DataTypeINT8 {
		if kCache.rawData != nil && len(kCache.rawData) >= off+size &&
			vCache.rawData != nil && len(vCache.rawData) >= off+size {
			tmpK := make([]int8, size)
			tmpV := make([]int8, size)
			QuantizeBlockQ8_0(tData[:size], tmpK)
			QuantizeBlockQ8_0(vData[:size], tmpV)
			for i := 0; i < size; i++ {
				kCache.rawData[off+i] = byte(tmpK[i]) // #nosec G115
				vCache.rawData[off+i] = byte(tmpV[i]) // #nosec G115
			}
			return
		}
	}

	// Fallback to standard StoreKV
	t.StoreKV(v, kCache, vCache, pos, heads, headDim, windowSize)
}

// VisionPatchEmbed performs patch embedding projection on CPU.
// Note: Full implementation requires SIMD-optimized GEMM; this is a reference implementation.
func (c *Context) VisionPatchEmbed(pixels *Tensor, weights *Tensor, output *Tensor, patchSize, visionDim, numPatchesX int) {
	if pixels == nil || weights == nil || output == nil {
		return
	}
	input := pixels.ToHostF32()
	wt := weights.ToHostF32()
	out := output.ToHostF32()

	hiddenSize := len(wt) / len(out)

	for i := 0; i < len(out); i++ {
		sum := float32(0)
		for j := 0; j < hiddenSize && i*hiddenSize+j < len(wt); j++ {
			sum += input[j] * wt[i*hiddenSize+j]
		}
		out[i] = sum
	}
	_ = output.LoadFrom(out)
}

// VisionPatchEmbedGemma4 performs Gemma 4 patch embedding on CPU.
func (c *Context) VisionPatchEmbedGemma4(pixels *Tensor, weights *Tensor, bias *Tensor, output *Tensor, patchSize, hiddenDim, numPatches int) {
	if pixels == nil || weights == nil || output == nil {
		return
	}

	input := pixels.ToHostF32()
	wt := weights.ToHostF32()
	out := output.ToHostF32()
	biasVals := make([]float32, 0)
	if bias != nil {
		biasVals = bias.ToHostF32()
	}

	inputSize := len(input) / numPatches

	for p := 0; p < numPatches; p++ {
		offset := p * hiddenDim
		for i := 0; i < hiddenDim && offset+i < len(out); i++ {
			sum := float32(0)
			for j := 0; j < inputSize; j++ {
				srcIdx := p*inputSize + j
				wIdx := i*inputSize + j
				if srcIdx < len(input) && wIdx < len(wt) {
					sum += input[srcIdx] * wt[wIdx]
				}
			}
			out[offset+i] = sum
			if len(biasVals) > i {
				out[offset+i] += biasVals[i]
			}
		}
	}
	_ = output.LoadFrom(out)
}

// ============================================================================
// MOE (Mixture of Experts) Operations
// ============================================================================

// MOERouterLogits computes routing logits for MOE layer
func (c *Context) MOERouterLogits(input, gateWeight *Tensor) *Tensor {
	batchSize := input.Rows()
	dim := input.Cols()
	numExperts := gateWeight.Rows()

	logits := c.NewTensorFP32(batchSize, numExperts)
	inData := input.ToHost()
	gateData := gateWeight.ToHost()
	logitData := make([]float32, batchSize*numExperts)

	for b := 0; b < batchSize; b++ {
		for e := 0; e < numExperts; e++ {
			var sum float32
			for d := 0; d < dim; d++ {
				sum += inData[b*dim+d] * gateData[e*dim+d]
			}
			logitData[b*numExperts+e] = sum
		}
	}
	_ = logits.LoadFrom(logitData)
	return logits
}

// MOETopKSelection selects top-k experts per token and computes softmax weights
func (c *Context) MOETopKSelection(logits *Tensor, topK int) (*Tensor, *Tensor) {
	batchSize := logits.Rows()
	numExperts := logits.Cols()

	expertIndices := c.NewTensorFP32(batchSize, topK)
	expertWeights := c.NewTensorFP32(batchSize, topK)
	logitData := logits.ToHost()

	indicesData := make([]float32, batchSize*topK)
	weightsData := make([]float32, batchSize*topK)

	for b := 0; b < batchSize; b++ {
		row := logitData[b*numExperts : (b+1)*numExperts]
		selected := make([]int, topK)
		selectedLogits := make([]float32, topK)

		for k := 0; k < topK; k++ {
			maxVal := float32(-1e30)
			maxIdx := -1
			for e := 0; e < numExperts; e++ {
				already := false
				for prev := 0; prev < k; prev++ {
					if selected[prev] == e {
						already = true
						break
					}
				}
				if already {
					continue
				}
				if row[e] > maxVal || (row[e] == maxVal && (maxIdx < 0 || e < maxIdx)) {
					maxVal = row[e]
					maxIdx = e
				}
			}
			selected[k] = maxIdx
			selectedLogits[k] = maxVal
		}

		maxLogit := float32(-1e30)
		for k := 0; k < topK; k++ {
			if selectedLogits[k] > maxLogit {
				maxLogit = selectedLogits[k]
			}
		}

		var sumExp float32
		expVals := make([]float32, topK)
		for k := 0; k < topK; k++ {
			expVals[k] = float32(math.Exp(float64(selectedLogits[k] - maxLogit)))
			sumExp += expVals[k]
		}
		invSum := float32(1.0 / (float64(sumExp) + 1e-9))
		for k := 0; k < topK; k++ {
			indicesData[b*topK+k] = float32(selected[k])
			weightsData[b*topK+k] = expVals[k] * invSum
		}
	}

	_ = expertIndices.LoadFrom(indicesData)
	_ = expertWeights.LoadFrom(weightsData)
	return expertIndices, expertWeights
}

// MOEExpertForward applies selected experts to input with weighted mixing
func (c *Context) MOEExpertForward(input, expertWeight, expertIndices, expertWeights *Tensor, hiddenDim int) *Tensor {
	batchSize := input.Rows()
	dim := input.Cols()
	topK := expertIndices.Cols()

	output := c.NewTensorFP32(batchSize, hiddenDim)
	inData := input.ToHost()
	wData := expertWeight.ToHost()
	idxData := expertIndices.ToHost()
	weightData := expertWeights.ToHost()
	outData := make([]float32, batchSize*hiddenDim)

	for b := 0; b < batchSize; b++ {
		for d := 0; d < hiddenDim; d++ {
			var sum float32
			for k := 0; k < topK; k++ {
				w := weightData[b*topK+k]
				if w == 0 {
					continue
				}
				expert := int(idxData[b*topK+k])
				if expert < 0 {
					continue
				}
				expertOffset := (expert*hiddenDim + d) * dim
				var dot float32
				for i := 0; i < dim; i++ {
					dot += inData[b*dim+i] * wData[expertOffset+i]
				}
				sum += w * dot
			}
			outData[b*hiddenDim+d] = sum
		}
	}
	_ = output.LoadFrom(outData)
	return output
}

// MOEExpertGateUpSwiGLU applies fused gate, up and SwiGLU forward pass for multiple experts
func (c *Context) MOEExpertGateUpSwiGLU(input, gateWeight, upWeight, expertIndices, expertWeights *Tensor, hiddenDim int) *Tensor {
	batchSize := input.Rows()
	dim := input.Cols()
	topK := expertIndices.Cols()

	output := c.NewTensorFP32(batchSize, hiddenDim)
	inData := input.ToHost()
	gateData := gateWeight.ToHost()
	upData := upWeight.ToHost()
	idxData := expertIndices.ToHost()
	weightData := expertWeights.ToHost()
	outData := make([]float32, batchSize*hiddenDim)

	for b := 0; b < batchSize; b++ {
		for h := 0; h < hiddenDim; h++ {
			var sumAct float32
			for k := 0; k < topK; k++ {
				w := weightData[b*topK+k]
				if w == 0 {
					continue
				}
				expert := int(idxData[b*topK+k])
				if expert < 0 {
					continue
				}
				expertRow := expert*hiddenDim + h
				var gateDot, upDot float32
				for i := 0; i < dim; i++ {
					inV := inData[b*dim+i]
					gateDot += inV * gateData[expertRow*dim+i]
					upDot += inV * upData[expertRow*dim+i]
				}
				gClamped := gateDot
				if gClamped < -15.0 {
					gClamped = -15.0
				} else if gClamped > 15.0 {
					gClamped = 15.0
				}
				siluGate := gateDot / (1.0 + float32(math.Exp(float64(-gClamped))))
				sumAct += w * (siluGate * upDot)
			}
			outData[b*hiddenDim+h] = sumAct
		}
	}
	_ = output.LoadFrom(outData)
	return output
}

// =============================================================================
// MLA (Multi-Head Latent Attention) Operations
// =============================================================================

// MLADecompressKV decompresses compressed latent KV cache into separate key (non-rotary) and value tensors
// compressedKV: [numTokens, kvLoraRank]
// wUKV: [heads * (qkNopeDim + vHeadDim), kvLoraRank]
// Returns: kNope [numTokens, heads * qkNopeDim], v [numTokens, heads * vHeadDim]
func (c *Context) MLADecompressKV(compressedKV, wUKV *Tensor, numTokens, kvLoraRank, heads, qkNopeDim, vHeadDim int) (*Tensor, *Tensor) {
	totalKRows := heads * qkNopeDim
	totalVRows := heads * vHeadDim
	kNope := c.NewTensorFP32(numTokens, totalKRows)
	v := c.NewTensorFP32(numTokens, totalVRows)

	kvData := compressedKV.ToHost()
	wData := wUKV.ToHost()
	kData := make([]float32, numTokens*totalKRows)
	vData := make([]float32, numTokens*totalVRows)

	for t := 0; t < numTokens; t++ {
		kvToken := kvData[t*kvLoraRank : (t+1)*kvLoraRank]
		for r := 0; r < totalKRows; r++ {
			wRow := wData[r*kvLoraRank : (r+1)*kvLoraRank]
			var sum float32
			for j := 0; j < kvLoraRank; j++ {
				sum += kvToken[j] * wRow[j]
			}
			kData[t*totalKRows+r] = sum
		}
		for r := 0; r < totalVRows; r++ {
			wRow := wData[(totalKRows+r)*kvLoraRank : (totalKRows+r+1)*kvLoraRank]
			var sum float32
			for j := 0; j < kvLoraRank; j++ {
				sum += kvToken[j] * wRow[j]
			}
			vData[t*totalVRows+r] = sum
		}
	}

	_ = kNope.LoadFrom(kData)
	_ = v.LoadFrom(vData)
	return kNope, v
}

// MLAProjectQuerySplitRoPE splits query projections into content and rotary parts, applying RoPE to rotary part
// qAll: [numTokens, heads * (qkNopeDim + qkRopeDim)]
// posIds: [numTokens] (optional int32 tensor)
// Returns: qNope [numTokens, heads * qkNopeDim], qRope [numTokens, heads * qkRopeDim]
func (c *Context) MLAProjectQuerySplitRoPE(qAll, posIds *Tensor, numTokens, heads, qkNopeDim, qkRopeDim int, theta float32) (*Tensor, *Tensor) {
	qNope := c.NewTensorFP32(numTokens, heads*qkNopeDim)
	qRope := c.NewTensorFP32(numTokens, heads*qkRopeDim)

	qAllData := qAll.ToHost()
	var posData []float32
	if posIds != nil {
		posData = posIds.ToHost()
	}

	qNopeData := make([]float32, numTokens*heads*qkNopeDim)
	qRopeData := make([]float32, numTokens*heads*qkRopeDim)
	inHeadDim := qkNopeDim + qkRopeDim
	halfRope := qkRopeDim / 2

	for t := 0; t < numTokens; t++ {
		pos := t
		if len(posData) > t {
			pos = int(posData[t])
		}
		for h := 0; h < heads; h++ {
			baseIn := (t*heads + h) * inHeadDim
			baseNope := (t*heads + h) * qkNopeDim
			baseRope := (t*heads + h) * qkRopeDim

			// Copy nope
			copy(qNopeData[baseNope:baseNope+qkNopeDim], qAllData[baseIn:baseIn+qkNopeDim])

			// Apply RoPE on rotary part
			for i := 0; i < halfRope; i++ {
				freq := float32(pos) * float32(math.Pow(float64(theta), float64(-2*i)/float64(qkRopeDim)))
				cosVal := float32(math.Cos(float64(freq)))
				sinVal := float32(math.Sin(float64(freq)))

				x0 := qAllData[baseIn+qkNopeDim+i]
				x1 := qAllData[baseIn+qkNopeDim+i+halfRope]

				qRopeData[baseRope+i] = x0*cosVal - x1*sinVal
				qRopeData[baseRope+i+halfRope] = x0*sinVal + x1*cosVal
			}
		}
	}

	_ = qNope.LoadFrom(qNopeData)
	_ = qRope.LoadFrom(qRopeData)
	return qNope, qRope
}

// MLAAbsorbedQuery projects content query into latent space per head using W_UK
// qNope: [numTokens, heads * qkNopeDim]
// wUK: [heads * qkNopeDim, kvLoraRank]
// Returns: qAbsorbed [numTokens, heads * kvLoraRank]
func (c *Context) MLAAbsorbedQuery(qNope, wUK *Tensor, numTokens, heads, qkNopeDim, kvLoraRank int) *Tensor {
	qAbsorbed := c.NewTensorFP32(numTokens, heads*kvLoraRank)
	qData := qNope.ToHost()
	wData := wUK.ToHost()
	absData := make([]float32, numTokens*heads*kvLoraRank)

	for t := 0; t < numTokens; t++ {
		for h := 0; h < heads; h++ {
			qHead := qData[(t*heads+h)*qkNopeDim : (t*heads+h+1)*qkNopeDim]
			for j := 0; j < kvLoraRank; j++ {
				var sum float32
				for k := 0; k < qkNopeDim; k++ {
					sum += qHead[k] * wData[(h*qkNopeDim+k)*kvLoraRank+j]
				}
				absData[(t*heads+h)*kvLoraRank+j] = sum
			}
		}
	}

	_ = qAbsorbed.LoadFrom(absData)
	return qAbsorbed
}

// MLAAbsorbedDecodeAttention executes fused decode attention directly over compressed KV cache
// qAbsorbed: [numTokens, heads * kvLoraRank]
// qRope: [numTokens, heads * qkRopeDim]
// kCache: [seqLen, kvLoraRank]
// kRopeCache: [seqLen, qkRopeDim]
// wUV: [heads * vHeadDim, kvLoraRank]
// Returns: output [numTokens, heads * vHeadDim]
func (c *Context) MLAAbsorbedDecodeAttention(qAbsorbed, qRope, kCache, kRopeCache, wUV *Tensor, numTokens, seqLen, heads, kvLoraRank, qkRopeDim, vHeadDim int, scale float32) *Tensor {
	output := c.NewTensorFP32(numTokens, heads*vHeadDim)
	qAbsData := qAbsorbed.ToHost()
	qRopeData := qRope.ToHost()
	kData := kCache.ToHost()
	kRopeData := kRopeCache.ToHost()
	wData := wUV.ToHost()
	outData := make([]float32, numTokens*heads*vHeadDim)

	scores := make([]float32, seqLen)
	latent := make([]float32, kvLoraRank)

	for t := 0; t < numTokens; t++ {
		for h := 0; h < heads; h++ {
			curQAbs := qAbsData[(t*heads+h)*kvLoraRank : (t*heads+h+1)*kvLoraRank]
			curQRope := qRopeData[(t*heads+h)*qkRopeDim : (t*heads+h+1)*qkRopeDim]

			maxScore := float32(-1e30)
			for s := 0; s < seqLen; s++ {
				curK := kData[s*kvLoraRank : (s+1)*kvLoraRank]
				curKRope := kRopeData[s*qkRopeDim : (s+1)*qkRopeDim]

				var dotC, dotR float32
				for j := 0; j < kvLoraRank; j++ {
					dotC += curQAbs[j] * curK[j]
				}
				for j := 0; j < qkRopeDim; j++ {
					dotR += curQRope[j] * curKRope[j]
				}
				sc := (dotC + dotR) * scale
				scores[s] = sc
				if sc > maxScore {
					maxScore = sc
				}
			}

			var sumExp float32
			for s := 0; s < seqLen; s++ {
				expVal := float32(math.Exp(float64(scores[s] - maxScore)))
				scores[s] = expVal
				sumExp += expVal
			}
			invSum := float32(1.0 / (float64(sumExp) + 1e-9))
			for s := 0; s < seqLen; s++ {
				scores[s] *= invSum
			}

			for j := 0; j < kvLoraRank; j++ {
				var sumLatent float32
				for s := 0; s < seqLen; s++ {
					sumLatent += scores[s] * kData[s*kvLoraRank+j]
				}
				latent[j] = sumLatent
			}

			wHead := wData[(h*vHeadDim)*kvLoraRank : ((h+1)*vHeadDim)*kvLoraRank]
			for i := 0; i < vHeadDim; i++ {
				wRow := wHead[i*kvLoraRank : (i+1)*kvLoraRank]
				var outVal float32
				for j := 0; j < kvLoraRank; j++ {
					outVal += latent[j] * wRow[j]
				}
				outData[(t*heads+h)*vHeadDim+i] = outVal
			}
		}
	}

	_ = output.LoadFrom(outData)
	return output
}
