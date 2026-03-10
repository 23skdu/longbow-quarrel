//go:build darwin && metal

package engine

import (
	"fmt"
	"sync"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func FuzzSharedKVCache(f *testing.F) {
	f.Add(uint(10), uint(5))
	f.Add(uint(50), uint(10))

	ctx := device.NewContext()
	defer ctx.Free()

	f.Fuzz(func(t *testing.T, ops uint, concurrency uint) {
		if ops > 50 {
			ops = 50
		}
		if concurrency > 5 {
			concurrency = 5
		}
		if concurrency == 0 {
			concurrency = 1
		}

		conf := config.Default()
		conf.KVHeads = 1
		conf.HeadDim = 32
		conf.Layers = 1
		conf.WindowSize = 256 // Capacity for a few sequences

		cache := &PagedKVCache{}
		if err := cache.Init(ctx, conf); err != nil {
			t.Fatalf("Failed to init cache: %v", err)
		}
		defer cache.Free()

		tensors := make([]*device.Tensor, concurrency*2)
		for i := uint(0); i < concurrency*2; i++ {
			tensors[i] = ctx.NewTensor(1, 32)
		}
		defer func() {
			for _, t := range tensors {
				t.Free()
			}
		}()

		var wg sync.WaitGroup

		for c := uint(0); c < concurrency; c++ {
			wg.Add(1)
			seqID := fmt.Sprintf("seq-%d", c)
			k := tensors[c*2]
			v := tensors[c*2+1]

			go func(seq string, ops uint, kc, vc *device.Tensor) {
				defer wg.Done()
				pos := 0
				for i := uint(0); i < ops; i++ {
					_ = cache.Update(seq, 0, pos, kc, vc)
					_ = cache.Get(seq, 0)
					pos++
				}
			}(seqID, ops, k, v)
		}
		wg.Wait()
	})
}
