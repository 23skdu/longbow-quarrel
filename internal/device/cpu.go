//go:build (!darwin || !metal) && (!linux || !cuda)

package device

import (
	"fmt"
	"runtime"
	"sync/atomic"
)

type Context struct {
	device     int
	memUsed    int64
	numThreads int
}

func NewContext() *Context {
	return &Context{
		device:     -1,
		memUsed:    0,
		numThreads: runtime.NumCPU(),
	}
}

func (c *Context) Device() int {
	return c.device
}

func (c *Context) Free() {
	c.memUsed = 0
}

func (c *Context) NewTensor(rows, cols int) *Tensor {
	return &Tensor{
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
func (c *Context) NewTensorWithType(rows, cols int, dt DataType) *Tensor {
	t := c.NewTensor(rows, cols)
	t.dataType = dt
	return t
}

type Tensor struct {
	data     []float32
	dims     []int
	strides  []int
	name     string
	dataType DataType
}

func NewTensor(name string, data []float32) *Tensor {
	dims := []int{len(data)}
	strides := []int{1}
	return &Tensor{
		data:    data,
		dims:    dims,
		strides: strides,
		name:    name,
	}
}

func (t *Tensor) Dims() []int {
	return t.dims
}

func (t *Tensor) Strides() []int {
	return t.strides
}

func (t *Tensor) Data() []float32 {
	return t.data
}

func (t *Tensor) Name() string {
	return t.name
}

func (t *Tensor) Free() {
	t.data = nil
}

func (t *Tensor) ZeroInit() {
	for i := range t.data {
		t.data[i] = 0
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
	// CPU implementation: Copy from t and v to kCache and vCache
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
