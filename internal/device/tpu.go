//go:build linux && tpu

package device

/*
#cgo linux,amd64 LDFLAGS: -L${SRCDIR} -llibtpu -lxla -ltensorflow
#cgo linux,amd64 CFLAGS: -I/usr/local/tpu/include -I${SRCDIR}
#include <libtpu/libtpu.h>
#include <xla/client/lib.h>
#include <xla/status.h>
#include <stdio.h>
#include <stdlib.h>

typedef enum {
    TPU_DTYPE_F32 = 0,
    TPU_DTYPE_F16 = 1,
    TPU_DTYPE_BF16 = 2,
    TPU_DTYPE_INT8 = 3,
} TPUDataType;

extern void tpuInitialize();
extern void tpuShutdown();
extern int tpuGetDeviceCount();
extern int tpuGetDefaultDevice();
extern void* tpuAllocate(size_t size);
extern void tpuFree(void* ptr);
extern void tpuMemcpyH2D(void* dst, const void* src, size_t size);
extern void tpuMemcpyD2H(void* dst, const void* src, size_t size);
extern void tpuMemset(void* ptr, int value, size_t size);
extern void tpuStreamCreate(int device, void** stream);
extern void tpuStreamDestroy(void* stream);
extern void tpuStreamSynchronize(void* stream);

// XLA computation kernels
extern void tpuRMSNorm(void* input, void* weight, void* output, int rows, int cols, float eps, void* stream);
extern void tpuMatmul(void* a, void* b, void* output, int m, int n, int k, void* stream);
extern void tpuFusedAttention(void* q, void* k, void* v, void* output, int batch, int heads, int seqLen, int kvSeqLen, int headDim, float scale, void* stream);
extern void tpuFusedRoPE(void* tensor, const int* posIds, int batch, int heads, int seqLen, int headDim, float theta, void* stream);
extern void tpuFusedMLP(void* input, void* gateWeight, void* upWeight, void* downWeight, void* output, int batch, int dim, int hiddenDim, void* stream);
extern void tpuStoreKV(void* k, void* v, void* kCache, void* vCache, const int* positions, int numTokens, void* stream);
extern void tpuAttentionPaged(void* q, void* kPool, void* vPool, void* output, const int* tokenPositions, const int* blockTables, const int* tokenToSeq, int maxBlocks, int heads, int kvHeads, int headDim, int blockSize, int numTokens, float scale, void* stream);
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

var (
	tpuInitialized bool
	tpuMu         sync.Mutex
)

type Context struct {
	Device   int
	Stream   unsafe.Pointer
	pool     *tensorPool
	xlaComp  unsafe.Pointer
}

func (ctx *Context) DeviceID() int {
	return ctx.Device
}

func InitTPU() error {
	tpuMu.Lock()
	defer tpuMu.Unlock()

	if tpuInitialized {
		return nil
	}

	C.tpuInitialize()
	tpuInitialized = true
	return nil
}

func ShutdownTPU() {
	tpuMu.Lock()
	defer tpuMu.Unlock()

	if tpuInitialized {
		C.tpuShutdown()
		tpuInitialized = false
	}
}

type Tensor struct {
	devPtr    unsafe.Pointer
	rows      int
	cols      int
	dataType DataType
	ctx      *Context
	pooled   bool
	sizeB   int
}

func NewTPUContext() (*Context, error) {
	if err := InitTPU(); err != nil {
		return nil, err
	}

	var stream unsafe.Pointer
	device := C.tpuGetDefaultDevice()
	C.tpuStreamCreate(C.int(device), &stream)

	return &Context{
		Device: int(device),
		Stream: stream,
		pool:   newTensorPool(),
	}, nil
}

func (ctx *Context) NewTensor(rows, cols int) *Tensor {
	size := rows * cols * 4
	devPtr := C.tpuAllocate(C.size_t(size))

	return &Tensor{
		devPtr:    devPtr,
		rows:     rows,
		cols:     cols,
		dataType: DataTypeF32,
		ctx:      ctx,
		pooled:   false,
		sizeB:   size,
	}
}

func (ctx *Context) NewQuantizedTensor(rows, cols int, qt QuantizationType) *Tensor {
	blockSize := qt.BlockSize()
	qblock := rows / blockSize
	size := cols * qt.BytesPerElement() * qblock
	devPtr := C.tpuAllocate(C.size_t(size))

	return &Tensor{
		devPtr:    devPtr,
		rows:     rows,
		cols:     cols,
		dataType: DataType(qt),
		ctx:      ctx,
		pooled:   false,
		sizeB:   size,
	}
}

func (t *Tensor) Free() {
	if t.devPtr != nil {
		C.tpuFree(t.devPtr)
		t.devPtr = nil
	}
}

func (t *Tensor) Rows() int      { return t.rows }
func (t *Tensor) Cols() int      { return t.cols }
func (t *Tensor) DataType() int { return int(t.dataType) }
func (t *Tensor) IsDevice() bool { return false }
func (t *Tensor) SizeBytes() int { return t.sizeB }

func (t *Tensor) LoadFrom(data interface{}) error {
	slice, err := toF32Slice(data)
	if err != nil {
		return err
	}

	size := len(slice) * 4
	C.tpuMemcpyH2D(t.devPtr, unsafe.Pointer(&slice[0]), C.size_t(size))
	return nil
}

func (t *Tensor) ToHostF32() ([]float32, error) {
	size := t.rows * t.cols
	result := make([]float32, size)
	C.tpuMemcpyD2H(unsafe.Pointer(&result[0]), t.devPtr, C.size_t(size*t.cols*4))
	return result, nil
}

func (t *Tensor) ToHostFP16() []uint16 {
	return nil
}

func (t *Tensor) ReturnToPool() {
	if t.pooled && t.ctx != nil && t.ctx.pool != nil {
		t.ctx.pool.Put(t)
		t.pooled = false
	}
}

func newTensorPool() *tensorPool {
	return &tensorPool{
		free: make(map[int][]*Tensor),
	}
}

func (p *tensorPool) Get(rows, cols int) *Tensor {
	key := rows*10000 + cols
	list, ok := p.free[key]
	if !ok || len(list) == 0 {
		return nil
	}

	n := len(list) - 1
	t := list[n]
	list[n] = nil
	p.free[key] = list[:n]
	return t
}

func (p *tensorPool) Put(t *Tensor) {
	key := t.rows*10000 + t.cols
	p.free[key] = append(p.free[key], t)
	t.pooled = true
}

type TPUModel struct {
	KCache []*Tensor
	VCache []*Tensor
	Weights map[string]*Tensor
	TokenEmb *Tensor
	Output   *Tensor
	Dim      int
	Layers   int
	Ctx      *Context
}

func (ctx *Context) LoadTPUModel(f *gguf.GGUFFile, lazy bool, kvCacheSize int) (*TPUModel, error) {
	m := &TPUModel{
		Weights: make(map[string]*Tensor),
		Ctx:      ctx,
	}

	dim := 0
	layers := 0
	if v, ok := f.KV["llama.embedding_length"].(uint32); ok {
		dim = int(v)
	}
	if v, ok := f.KV["llama.block_count"].(uint32); ok {
		layers = int(v)
	}

	m.Dim = dim
	m.Layers = layers

	tokenEmb, err := ctx.loadWeight(f, "token_embd.weight")
	if err == nil {
		m.TokenEmb = tokenEmb
	}

	output, err := ctx.loadWeight(f, "output.weight")
	if err != nil {
		output, _ = ctx.loadWeight(f, "token_embd.weight")
	}
	m.Output = output

	for i := 0; i < layers; i++ {
		m.KCache = append(m.KCache, ctx.NewTensor(1, dim))
		m.VCache = append(m.VCache, ctx.NewTensor(1, dim))
	}

	return m, nil
}

func (ctx *Context) loadWeight(f *gguf.GGUFFile, name string) (*Tensor, error) {
	name += ".weight"

	weight, ok := f.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("weight not found: %s", name)
	}

	t := ctx.NewTensor(weight.Dimensions[0], weight.Dimensions[1])
	if err := t.LoadFrom(weight.Data); err != nil {
		t.Free()
		return nil, fmt.Errorf("failed to load weight %s: %w", name, err)
	}

	return t, nil
}

func (ctx *Context) GetWeightTensor(name string) *Tensor {
	return nil
}

func (ctx *Context) MatmulF16(a, b *Tensor) (*Tensor, error) {
	if a.cols != b.rows {
		return nil, fmt.Errorf("matmul dimension mismatch: %d != %d", a.cols, b.rows)
	}

	out := ctx.NewTensor(a.rows, b.cols)
	C.tpuMatmul(a.devPtr, b.devPtr, out.devPtr,
		C.int(a.rows), C.int(b.cols), C.int(a.cols),
		ctx.Stream)

	return out, nil
}

func (ctx *Context) RMSNorm(input, weight, output *Tensor, rows, cols int, eps float32) {
	C.tpuRMSNorm(input.devPtr, weight.devPtr, output.devPtr,
		C.int(rows), C.int(cols), C.float(eps),
		ctx.Stream)
}

func (ctx *Context) FusedAttention(q, k, v, output, kCache, vCache *Tensor, batch, heads, seqLen, kvSeqLen, headDim int, scale float32, windowSize int) {
	C.tpuFusedAttention(q.devPtr, k.devPtr, v.devPtr, output.devPtr,
		C.int(batch), C.int(heads), C.int(seqLen), C.int(kvSeqLen), C.int(headDim), C.float(scale),
		ctx.Stream)
}

func (ctx *Context) FusedRoPE(q *Tensor, positions []int, batch, heads, seqLen, headDim int, theta float32) {
	posSlice := make([]C.int, len(positions))
	for i, p := range positions {
		posSlice[i] = C.int(p)
	}

	C.tpuFusedRoPE(q.devPtr, &posSlice[0],
		C.int(batch), C.int(heads), C.int(seqLen), C.int(headDim), C.float(theta),
		ctx.Stream)
}

func (ctx *Context) FusedMLP(input, gateW, upW, downW, output *Tensor, batch, dim, hiddenDim int) {
	C.tpuFusedMLP(input.devPtr, gateW.devPtr, upW.devPtr, downW.devPtr, output.devPtr,
		C.int(batch), C.int(dim), C.int(hiddenDim),
		ctx.Stream)
}

func (ctx *Context) StoreKV(k, v, kCache, vCache *Tensor, positions []int, numTokens int) {
	posSlice := make([]C.int, len(positions))
	for i, p := range positions {
		posSlice[i] = C.int(p)
	}

	C.tpuStoreKV(k.devPtr, v.devPtr, kCache.devPtr, vCache.devPtr,
		&posSlice[0], C.int(numTokens), ctx.Stream)
}

func (ctx *Context) AttentionPaged(q, kPool, vPool, output, tokenPositions, blockTables, tokenToSeq *Tensor, maxBlocks, heads, kvHeads, headDim, blockSize, numTokens int, scale float32) {
	posData, _ := tokenPositions.ToHostF32()
	btData, _ := blockTables.ToHostF32()
	tsData, _ := tokenToSeq.ToHostF32()

	posSlice := make([]C.int, len(posData))
	btSlice := make([]C.int, len(btData))
	tsSlice := make([]C.int, len(tsData))

	for i, p := range posData {
		posSlice[i] = C.int(p)
	}
	for i, b := range btData {
		btSlice[i] = C.int(b)
	}
	for i, t := range tsData {
		tsSlice[i] = C.int(t)
	}

	C.tpuAttentionPaged(q.devPtr, kPool.devPtr, vPool.devPtr, output.devPtr,
		&posSlice[0], &btSlice[0], &tsSlice[0],
		C.int(maxBlocks), C.int(heads), C.int(kvHeads), C.int(headDim), C.int(blockSize), C.int(numTokens), C.float(scale),
		ctx.Stream)
}

func (ctx *Context) Synchronize() {
	C.tpuStreamSynchronize(ctx.Stream)
}

func (ctx *Context) Free() {
	if ctx.Stream != nil {
		C.tpuStreamDestroy(ctx.Stream)
		ctx.Stream = nil
	}

	if ctx.pool != nil {
		for _, tensors := range ctx.pool.free {
			for _, t := range tensors {
				t.Free()
			}
		}
	}
}

func TPUAllocatedBytes() uint64 {
	return 0
}

func TPUDeviceCount() int {
	if !tpuInitialized {
		return 0
	}
	return int(C.tpuGetDeviceCount())
}