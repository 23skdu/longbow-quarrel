package engine

import "github.com/23skdu/longbow-quarrel/internal/device"

// poolValue reads the value the KV pool holds at the given element offset.
//
// The pool's declared element type now matches what the backend actually
// stores in it: float32 on the CPU, packed half-precision values on the
// GPU backends. The device layer converts to and from float32 on the host
// side, so the read is the same everywhere.
//
// It used to be backend-specific: the GPU backends stored half-precision
// values in a pool declared float32, so the read had to reach into the
// raw bytes and decode the halves by hand.
func poolValue(pool *device.Tensor, off int) float32 {
	return pool.ToHostF32()[off]
}
