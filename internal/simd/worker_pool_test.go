package simd

import (
	"sync"
	"sync/atomic"
	"testing"
)

func TestWorkerPool_Basic(t *testing.T) {
	pool := NewWorkerPool(4)
	defer pool.Close()

	n := 1000
	data := make([]int, n)
	pool.ParallelFor(n, func(start, end int) {
		for i := start; i < end; i++ {
			data[i] = i * 2
		}
	})

	for i := 0; i < n; i++ {
		if data[i] != i*2 {
			t.Fatalf("mismatch at %d: expected %d, got %d", i, i*2, data[i])
		}
	}
}

func TestWorkerPool_EdgeCases(t *testing.T) {
	pool := NewWorkerPool(4)
	defer pool.Close()

	// 0 elements
	pool.ParallelFor(0, func(start, end int) {
		t.Fatal("should not be called for 0 elements")
	})

	// 1 element
	called := false
	pool.ParallelFor(1, func(start, end int) {
		if start != 0 || end != 1 {
			t.Fatalf("unexpected range: [%d, %d)", start, end)
		}
		called = true
	})
	if !called {
		t.Fatal("expected 1 element to be processed")
	}

	// nil func
	pool.ParallelFor(10, nil)

	// total < workers
	count := int32(0)
	pool.ParallelFor(2, func(start, end int) {
		for i := start; i < end; i++ {
			atomic.AddInt32(&count, 1)
		}
	})
	if count != 2 {
		t.Fatalf("expected 2 processed, got %d", count)
	}
}

func TestWorkerPool_ConcurrentCalls(t *testing.T) {
	pool := GetWorkerPool()

	numGoroutines := 30
	elementsPerGoroutine := 500

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func(gid int) {
			defer wg.Done()
			data := make([]int, elementsPerGoroutine)
			pool.ParallelFor(elementsPerGoroutine, func(start, end int) {
				for i := start; i < end; i++ {
					data[i] = gid + i
				}
			})
			for i := 0; i < elementsPerGoroutine; i++ {
				if data[i] != gid+i {
					t.Errorf("goroutine %d mismatch at index %d: expected %d, got %d", gid, i, gid+i, data[i])
					return
				}
			}
		}(g)
	}

	wg.Wait()
}

func TestWorkerPool_GlobalParallelFor(t *testing.T) {
	n := 128
	sum := int64(0)
	ParallelFor(n, func(start, end int) {
		var localSum int64
		for i := start; i < end; i++ {
			localSum += int64(i)
		}
		atomic.AddInt64(&sum, localSum)
	})

	expected := int64((n - 1) * n / 2)
	if sum != expected {
		t.Fatalf("expected sum %d, got %d", expected, sum)
	}
}

func BenchmarkWorkerPool_ParallelFor(b *testing.B) {
	n := 1024
	data := make([]int, n)
	pool := GetWorkerPool()
	b.ReportAllocs()
	for b.Loop() {
		pool.ParallelFor(n, func(start, end int) {
			for j := start; j < end; j++ {
				data[j] = j * 2
			}
		})
	}
}

func BenchmarkWorkerPool_RawGoroutines(b *testing.B) {
	n := 1024
	data := make([]int, n)
	workers := 8
	chunkSize := (n + workers - 1) / workers
	b.ReportAllocs()
	for b.Loop() {
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			start := w * chunkSize
			end := start + chunkSize
			if end > n {
				end = n
			}
			if start >= end {
				continue
			}
			wg.Add(1)
			go func(s, e int) {
				defer wg.Done()
				for j := s; j < e; j++ {
					data[j] = j * 2
				}
			}(start, end)
		}
		wg.Wait()
	}
}

