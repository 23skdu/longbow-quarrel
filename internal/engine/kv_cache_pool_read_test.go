//go:build !cuda && !metal && !tpu

package engine

import "github.com/23skdu/longbow-quarrel/internal/device"

// poolValue reads the value the KV pool holds at the given element offset.
//
// The CPU backend stores the pool as float32, so the value is read
// directly.
func poolValue(pool *device.Tensor, off int) float32 {
	return pool.ToHostF32()[off]
}
