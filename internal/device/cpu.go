//go:build (linux && !cuda) || darwin

package device

import (
	"math"
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

type Tensor struct {
	data    []float32
	dims    []int
	strides []int
	name    string
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

type ActivationStats struct {
	Max    float32
	Min    float32
	Mean   float32
	RMS    float32
	Zeros  int
	NaNs   int
	Infs   int
	Sample []float32
}

func ComputeActivationStats(data []float32, sampleSize int) ActivationStats {
	stats := ActivationStats{
		Sample: make([]float32, 0),
	}

	for _, v := range data {
		if v > stats.Max {
			stats.Max = v
		}
		if v < stats.Min || stats.Min == 0 {
			stats.Min = v
		}
		stats.Mean += v
		stats.RMS += v * v

		if math.IsNaN(float64(v)) {
			stats.NaNs++
		}
		if math.IsInf(float64(v), 0) {
			stats.Infs++
		}
	}

	n := float32(len(data))
	stats.Mean /= n
	stats.RMS = float32(math.Sqrt(float64(stats.RMS / n)))

	if len(data) > 0 && sampleSize > 0 {
		step := len(data) / sampleSize
		for i := 0; i < sampleSize && i*step < len(data); i++ {
			stats.Sample = append(stats.Sample, data[i*step])
		}
	}

	return stats
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
