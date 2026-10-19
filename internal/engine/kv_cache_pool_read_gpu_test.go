//go:build cuda && !metal && !tpu

package engine

import (
	"encoding/binary"
	"math"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

// poolValue reads the value the KV pool holds at the given element offset.
//
// The CUDA backend writes the pool through a kernel that stores
// half-precision values packed two bytes per element, even though the pool
// tensor is declared float32. The store kernel and the attention kernel agree
// on that layout, so the cache works, but a host-side read has to interpret
// the bytes the same way.
//
// The pool holds half values, and the element at the given offset is the half
// value whose 16 bits sit at byte offset 2*off. That offset is always even,
// so the two bytes never straddle a float32 word.
func poolValue(pool *device.Tensor, off int) float32 {
	raw := pool.ToHostF32()
	byteOff := 2 * off
	word := byteOff / 4
	shift := byteOff % 4

	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], math.Float32bits(raw[word]))
	half := binary.LittleEndian.Uint16(b[shift : shift+2])

	return device.Float16ToFloat32(half)
}
