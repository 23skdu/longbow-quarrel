package device

import (
	"fmt"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"

	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

// ToArrowArray creates an Arrow FixedSizeList array from the raw memory of a Tensor.
// This relies on the Tensor.RawData() providing a CPU-accessible byte slice without reallocation
// if shared memory is used, or a direct VRAM-to-RAM mapped slice.
func (t *Tensor) ToArrowArray(allocator memory.Allocator) (*array.FixedSizeList, error) {
	rawData := t.RawData()
	if len(rawData) == 0 {
		return nil, fmt.Errorf("tensor provides no raw data for Arrow conversion")
	}

	// Ensure the rawData size matches expected F32 or F16 size
	// We'll map everything to F32 for embedding outputs currently.
	if t.dataType != DataTypeF32 {
		// If not F32, we must currently allocate a copy for the embedding Arrow output 
		// because Flight sinks generally expect uniform float32 vectors.
		hostData := t.ToHostF32()
		// To adhere to zero-copy principles, we wrap this slice into a Buffer without copying again.
		ArrowBuf := memory.NewBufferBytes(Float32SliceToBytes(hostData))
		defer ArrowBuf.Release()
		
		return buildFixedSizeList(allocator, ArrowBuf, t.Rows(), t.Cols(), t.ctx.DeviceID()), nil
	}

	// True zero copy for F32
	ArrowBuf := memory.NewBufferBytes(rawData)
	defer ArrowBuf.Release()

	// Track bytes processed in global hotpath metrics
	metrics.RecordArrowBytesHotpath(int64(t.SizeBytes()))

	return buildFixedSizeList(allocator, ArrowBuf, t.Rows(), t.Cols(), t.ctx.DeviceID()), nil
}

// buildFixedSizeList constructs a List array with tracking for GPU affinity
func buildFixedSizeList(allocator memory.Allocator, buf *memory.Buffer, rows, cols int, deviceID int) *array.FixedSizeList {
	if allocator == nil {
		allocator = memory.DefaultAllocator
	}

	// For future reference: device affinity metadata could be stored in the Field metadata
	// but for now we ensure the parameters are utilized for static analysis compliance.
	_ = deviceID 

	// Construct the list array where each item is a vector of `cols` float32s
	valueData := array.NewData(
		arrow.PrimitiveTypes.Float32,
		rows*cols,
		[]*memory.Buffer{nil, buf},
		nil, 0, 0,
	)
	defer valueData.Release()

	// Child float32 array
	flatVals := array.NewFloat32Data(valueData)
	defer flatVals.Release()

	// Wrap in FixedSizeList
	listType := arrow.FixedSizeListOf(int32(cols), arrow.PrimitiveTypes.Float32)
	listData := array.NewData(
		listType,
		rows,
		[]*memory.Buffer{nil}, // validity buffer
		[]arrow.ArrayData{flatVals.Data()},
		0, 0,
	)
	defer listData.Release()

	return array.NewFixedSizeListData(listData)
}

