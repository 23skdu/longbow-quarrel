package device

import (
	"fmt"
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
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
		ArrowBuf := memory.NewBufferBytes(float32SliceToBytes(hostData))
		defer ArrowBuf.Release()
		
		return buildFixedSizeList(allocator, ArrowBuf, t.Rows(), t.Cols()), nil
	}

	// True zero copy for F32
	ArrowBuf := memory.NewBufferBytes(rawData)
	defer ArrowBuf.Release()

	return buildFixedSizeList(allocator, ArrowBuf, t.Rows(), t.Cols()), nil
}

func buildFixedSizeList(allocator memory.Allocator, buf *memory.Buffer, rows, cols int) *array.FixedSizeList {
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

func float32SliceToBytes(s []float32) []byte {
    if len(s) == 0 {
        return nil
    }
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*4)
}
