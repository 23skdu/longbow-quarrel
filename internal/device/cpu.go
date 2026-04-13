//go:build (!darwin || !metal) && (!linux || !cuda)

package device

import (
	"fmt"
	"math"
	"runtime"
	"sync/atomic"

	"unsafe"

	"github.com/23skdu/longbow-quarrel/internal/simd"
)

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

func (c *Context) NewTensor(rows, cols int) *Tensor {
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

func (t *Tensor) Dims() []int {
	return t.dims
}

func (t *Tensor) Strides() []int {
	return t.strides
}

func (t *Tensor) RawData() []byte {
	if t.dataType == DataTypeF32 && t.data != nil {
		return unsafe.Slice((*byte)(unsafe.Pointer(&t.data[0])), len(t.data)*4)
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

func (t *Tensor) LoadFromF32(data []float32) {
	t.LoadFrom(data)
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
		q[i] = int8(src[i])
	}
	qj := make([]int8, qjlRows)
	for i := 0; i < qjlRows; i++ {
		qj[i] = int8(src[blockSize+i])
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

func (t *Tensor) LoadFrom(data []float32) error {
	if len(data) != len(t.data) {
		return fmt.Errorf("LoadFrom: size mismatch: %d != %d", len(data), len(t.data))
	}
	copy(t.data, data)
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
				dst[i] = byte(val)
			}
			for i, val := range qj {
				dst[blockSize+i] = byte(val)
			}
			setFloat32(dst[blockSize+qjlRows:blockSize+qjlRows+4], s)
			setFloat32(dst[blockSize+qjlRows+4:blockSize+qjlRows+8], sj)

			// Encode V
			vHeadData := v.data[headStart : headStart+headDim]
			qv, sv, resv := simd.PolarQuantSIMD(vHeadData, rot.data, blockSize, bits)
			qjv, sjv := simd.QJLTransformSIMD(resv, qjl.data, qjlRows, blockSize)

			vdst := vCache.rawData[blockCacheStart:]
			for i, val := range qv {
				vdst[i] = byte(val)
			}
			for i, val := range qjv {
				vdst[blockSize+i] = byte(val)
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
				dst[i] = byte(v)
			}
			for i, v := range qj {
				dst[blockSize+i] = byte(v)
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
				q[i] = int8(src[i])
			}
			qj = make([]int8, qjlRows)
			for i := 0; i < qjlRows; i++ {
				qj[i] = int8(src[blockSize+i])
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
		// 3. Add QJL contribution (Residual) already in original space
		if sj > 0 && qjlMatrix != nil {
			for i := 0; i < qjlRows; i++ {
				scale_i := float32(int8(qj[i])) * sj
				for j := 0; j < blockSize; j++ {
					out[j] += scale_i * qjlMatrix.data[i*blockSize+j]
				}
			}
		}
	}
}

func getFloat32(b []byte) float32 {
	bits := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
	return math.Float32frombits(bits)
}

func setFloat32(b []byte, f float32) {
	bits := math.Float32bits(f)
	b[0] = byte(bits)
	b[1] = byte(bits >> 8)
	b[2] = byte(bits >> 16)
	b[3] = byte(bits >> 24)
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

func RecordMemory(n int64) {
	atomic.AddInt64(&cpuAllocatedBytes, n)
}

func (c *Context) SetNumThreads(n int) {
	c.numThreads = n
}

func (c *Context) NumThreads() int {
	return c.numThreads
}
