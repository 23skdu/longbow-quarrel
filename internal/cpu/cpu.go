package cpu

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

var allocatedBytes int64

func traceAlloc(ptr *Tensor, delta int64, label string) {
	newVal := atomic.AddInt64(&allocatedBytes, delta)
	metrics.RecordGPUMemory(newVal)
}

func AllocatedBytes() int64 {
	return atomic.LoadInt64(&allocatedBytes)
}

var MaxMemory int64 = 32 * 1024 * 1024 * 1024

type Context struct {
	mu   sync.Mutex
	pool map[string][]*Tensor
}

func NewContext() *Context {
	return &Context{
		pool: make(map[string][]*Tensor),
	}
}

func (c *Context) Free() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, tensors := range c.pool {
		for _, t := range tensors {
			if t.data != nil {
				traceAlloc(t, -int64(capacity(t.data)), "free")
			}
		}
	}
	c.pool = make(map[string][]*Tensor)
}

type Tensor struct {
	data     interface{}
	shape    [2]int
	elemSize int
}

func capacity(d interface{}) int {
	switch v := d.(type) {
	case []byte:
		return cap(v)
	case []uint16:
		return cap(v)
	case []float32:
		return cap(v)
	case []float64:
		return cap(v)
	default:
		return 0
	}
}

func (c *Context) NewTensor(shape [2]int, elemSize int) *Tensor {
	key := shapeKey(shape, elemSize)
	c.mu.Lock()
	defer c.mu.Unlock()
	pool := c.pool[key]
	if len(pool) > 0 {
		t := pool[len(pool)-1]
		c.pool[key] = pool[:len(pool)-1]
		return t
	}
	var data interface{}
	size := shape[0] * shape[1]
	switch elemSize {
	case 1:
		data = make([]byte, size)
	case 2:
		data = make([]uint16, size)
	case 4:
		data = make([]float32, size)
	case 8:
		data = make([]float64, size)
	default:
		data = make([]byte, size)
	}
	traceAlloc(&Tensor{data: data}, int64(shape[0]*shape[1]*elemSize), "alloc")
	return &Tensor{
		data:     data,
		shape:    shape,
		elemSize: elemSize,
	}
}

func (c *Context) GetTensor(shape [2]int, elemSize int) *Tensor {
	return c.NewTensor(shape, elemSize)
}

func (c *Context) PutTensor(t *Tensor) {
	if t == nil || t.data == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	key := shapeKey(t.shape, t.elemSize)
	c.pool[key] = append(c.pool[key], t)
}

func shapeKey(shape [2]int, elemSize int) string {
	return string(rune(shape[0])) + string(rune(shape[1])) + string(rune(elemSize))
}

func (c *Context) SoftmaxF32(input *Tensor, output *Tensor) {
	inData := input.data.([]float32)
	outData := output.data.([]float32)
	copy(outData, inData)
	Softmax(outData)
}

func Softmax(x []float32) {
	if len(x) == 0 {
		return
	}
	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	sum := float32(0.0)
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i] - max)))
		sum += x[i]
	}
	if sum > 0 {
		invSum := float32(1.0) / sum
		for i := range x {
			x[i] *= invSum
		}
	}
}

func (c *Context) SwiGLU(gate, up, out *Tensor) {
	g := gate.data.([]float32)
	u := up.data.([]float32)
	o := out.data.([]float32)
	n := len(g)
	if n != len(u) || n != len(o) {
		return
	}
	for i := 0; i < n; i++ {
		val := g[i]
		if val > 10.0 {
			val = 10.0
		}
		if val < -10.0 {
			val = -10.0
		}
		sigmoid := float32(1.0) / (float32(1.0) + float32(math.Exp(float64(-val))))
		o[i] = u[i] * val * sigmoid
	}
}

func (c *Context) Fp16ToFp32(src, dst *Tensor) {
	Fp16ToFp32Func(src.data.([]uint16), dst.data.([]float32))
}

func Fp16ToFp32Func(src []uint16, dst []float32) {
	n := len(src)
	if n != len(dst) {
		return
	}
	for i := 0; i < n; i++ {
		h := src[i]
		sign := uint32(h>>15) & 0x1
		exp := uint32(h>>10) & 0x1F
		mant := uint32(h) & 0x3FF
		var f32 uint32
		if exp == 0 {
			if mant == 0 {
				f32 = sign << 31
			} else {
				shift := uint32(0)
				m := mant
				for m < 0x400 {
					m <<= 1
					shift++
				}
				m = (m & 0x3FF) << 13
				e := uint32(127 - 14 - shift)
				f32 = (sign << 31) | (e << 23) | m
			}
		} else if exp == 31 {
			if mant == 0 {
				f32 = (sign << 31) | 0x7F800000
			} else {
				f32 = (sign << 31) | 0x7F800000 | (mant << 13)
			}
		} else {
			newExp := exp - 15 + 127
			f32 = (sign << 31) | (newExp << 23) | (mant << 13)
		}
		dst[i] = math.Float32frombits(f32)
	}
}

func (c *Context) Fp32ToFp16(src, dst *Tensor) {
	Fp32ToFp16Func(src.data.([]float32), dst.data.([]uint16))
}

func Fp32ToFp16Func(src []float32, dst []uint16) {
	n := len(src)
	if n != len(dst) {
		return
	}
	for i := 0; i < n; i++ {
		f := src[i]
		bits := math.Float32bits(f)
		sign := bits >> 31
		exp := (bits >> 23) & 0xFF
		mant := bits & 0x7FFFFF
		var h uint16
		if exp == 0 {
			h = 0
		} else if exp == 255 {
			h = uint16(sign<<15) | 0x7C00 | uint16(mant>>9)
		} else {
			newExp := int(exp) - 127 + 15
			if newExp >= 31 {
				h = uint16(sign<<15) | 0x7C00
			} else if newExp <= 0 {
				shift := uint32(1 - newExp)
				m := mant | 0x800000
				h = uint16(sign<<15) | uint16(m>>(9+shift))
			} else {
				h = uint16(sign<<15) | uint16(newExp<<10) | uint16(mant>>13)
			}
		}
		dst[i] = h
	}
}

func (c *Context) LinearF32(weight *Tensor, input *Tensor, output *Tensor) {
	w := weight.data.([]float32)
	in := input.data.([]float32)
	out := output.data.([]float32)
	outRows := output.shape[0]
	outCols := output.shape[1]
	inCols := input.shape[1]
	parallelism := runtime.NumCPU()
	chunkSize := (outRows + parallelism - 1) / parallelism
	var wg sync.WaitGroup
	for i := 0; i < outRows; i += chunkSize {
		end := i + chunkSize
		if end > outRows {
			end = outRows
		}
		wg.Add(1)
		go func(rowStart, rowEnd int) {
			defer wg.Done()
			for row := rowStart; row < rowEnd; row++ {
				rowOffset := row * outCols
				inRowOffset := row * inCols
				for col := 0; col < outCols; col++ {
					var sum float32
					for k := 0; k < inCols; k++ {
						sum += in[inRowOffset+k] * w[k*outCols+col]
					}
					out[rowOffset+col] = sum
				}
			}
		}(i, end)
	}
	wg.Wait()
}

func (c *Context) RMSNorm(input *Tensor, weight *Tensor, output *Tensor, eps float32) {
	in := input.data.([]float32)
	w := weight.data.([]float32)
	out := output.data.([]float32)
	size := input.shape[1]
	numRows := input.shape[0]
	parallelism := runtime.NumCPU()
	chunkSize := (numRows + parallelism - 1) / parallelism
	var wg sync.WaitGroup
	for i := 0; i < numRows; i += chunkSize {
		end := i + chunkSize
		if end > numRows {
			end = numRows
		}
		wg.Add(1)
		go func(rowStart, rowEnd int) {
			defer wg.Done()
			for row := rowStart; row < rowEnd; row++ {
				rowOffset := row * size
				var sum float32
				for j := 0; j < size; j++ {
					v := in[rowOffset+j]
					sum += v * v
				}
				sum = float32(1.0) / float32(math.Sqrt(float64(sum/float32(size))+float64(eps)))
				for j := 0; j < size; j++ {
					out[rowOffset+j] = in[rowOffset+j] * sum * w[j]
				}
			}
		}(i, end)
	}
	wg.Wait()
}

func (c *Context) Rope(inout *Tensor, pos int, headDim int, theta float32) {
	data := inout.data.([]float32)
	numRows := inout.shape[0]
	seqLen := inout.shape[1]
	for row := 0; row < numRows; row++ {
		for h := 0; h < headDim; h += 2 {
			idx := row*seqLen*headDim + pos*headDim + h
			freq := float32(math.Pow(float64(theta), float64(h)/float64(headDim)))
			cosVal := float32(math.Cos(float64(freq)))
			sinVal := float32(math.Sin(float64(freq)))
			x0 := data[idx]
			x1 := data[idx+1]
			data[idx] = x0*cosVal - x1*sinVal
			data[idx+1] = x0*sinVal + x1*cosVal
		}
	}
}

func (c *Context) MatMul(a, b, out *Tensor) {
	parallelism := runtime.NumCPU()
	chunkSize := (out.shape[0] + parallelism - 1) / parallelism
	var wg sync.WaitGroup
	for i := 0; i < out.shape[0]; i += chunkSize {
		end := i + chunkSize
		if end > out.shape[0] {
			end = out.shape[0]
		}
		wg.Add(1)
		go func(rowStart, rowEnd int) {
			defer wg.Done()
			for row := rowStart; row < rowEnd; row++ {
				for col := 0; col < out.shape[1]; col++ {
					var sum float32
					for k := 0; k < a.shape[1]; k++ {
						sum += a.data.([]float32)[row*a.shape[1]+k] * b.data.([]float32)[k*b.shape[1]+col]
					}
					out.data.([]float32)[row*out.shape[1]+col] = sum
				}
			}
		}(i, end)
	}
	wg.Wait()
}

func (c *Context) AttentionQKV(q, k, v, out *Tensor, scale float32) {
}

func (c *Context) Add(a, b *Tensor) {
	aData := a.data.([]float32)
	bData := b.data.([]float32)
	for i := range aData {
		aData[i] += bData[i]
	}
}

func (c *Context) MulScalar(a *Tensor, s float32) {
	aData := a.data.([]float32)
	for i := range aData {
		aData[i] *= s
	}
}

func (c *Context) Silu(input, out *Tensor) {
	in := input.data.([]float32)
	o := out.data.([]float32)
	for i := range in {
		o[i] = in[i] / (float32(1.0) + float32(math.Exp(float64(-in[i]))))
	}
}

func (c *Context) GeLU(input, out *Tensor) {
	in := input.data.([]float32)
	o := out.data.([]float32)
	for i := range in {
		x := in[i]
		sqrtArg := x * float32(0.7978845608) * (float32(1.0) + float32(0.044715)*x*x)
		o[i] = float32(0.5) * x * (float32(1.0) + float32(math.Tanh(float64(sqrtArg))))
	}
}

func (c *Context) Embedding(weight *Tensor, ids []int, out *Tensor) {
	w := weight.data.([]float32)
	o := out.data.([]float32)
	embDim := weight.shape[1]
	for i, id := range ids {
		copy(o[i*embDim:(i+1)*embDim], w[id*embDim:(id+1)*embDim])
	}
}

func (c *Context) Slice(src *Tensor, dst *Tensor, srcStart [2]int, dstStart [2]int, size [2]int) {
	switch src.elemSize {
	case 4:
		srcData := src.data.([]float32)
		dstData := dst.data.([]float32)
		for i := 0; i < size[0]; i++ {
			for j := 0; j < size[1]; j++ {
				dstData[(dstStart[0]+i)*dst.shape[1]+dstStart[1]+j] = srcData[(srcStart[0]+i)*src.shape[1]+srcStart[1]+j]
			}
		}
	}
}

func (c *Context) Concat(a, b, out *Tensor, axis int) {
	switch axis {
	case 0:
		copy(out.data.([]float32), a.data.([]float32))
		copy(out.data.([]float32)[a.shape[0]*a.shape[1]:], b.data.([]float32))
	case 1:
		for row := 0; row < a.shape[0]; row++ {
			rowOffset := row * a.shape[1]
			dstOffset := row * out.shape[1]
			copy(out.data.([]float32)[dstOffset:], a.data.([]float32)[rowOffset:rowOffset+a.shape[1]])
			copy(out.data.([]float32)[dstOffset+a.shape[1]:], b.data.([]float32)[rowOffset:rowOffset+b.shape[1]])
		}
	}
}

func (c *Context) ViewAsTensor(data interface{}, shape [2]int, elemSize int) *Tensor {
	return &Tensor{data: data, shape: shape, elemSize: elemSize}
}

func (c *Context) GetTensorShape(t *Tensor) [2]int {
	return t.shape
}

func (c *Context) GetTensorElemSize(t *Tensor) int {
	return t.elemSize
}

func (c *Context) GetTensorData(t *Tensor) interface{} {
	return t.data
}
