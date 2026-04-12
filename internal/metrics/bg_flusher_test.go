package metrics

import (
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestBgFlusher_Unit(t *testing.T) {
	// Reset counters
	arrowBytesHotpath.Store(0)
	arrowEmbeddingsHotpath.Store(0)

	flusher := NewBgFlusher(10 * time.Millisecond)
	defer flusher.Stop()

	// Add hotpath metrics
	RecordArrowBytesHotpath(1024)
	RecordArrowEmbeddingHotpath()
	RecordArrowEmbeddingHotpath()

	// Wait for background flush
	time.Sleep(50 * time.Millisecond)

	// Check if atomic values were reset
	if b := arrowBytesHotpath.Load(); b != 0 {
		t.Errorf("Expected atomic bytes to be reset to 0, got %d", b)
	}

	if e := arrowEmbeddingsHotpath.Load(); e != 0 {
		t.Errorf("Expected atomic embeddings to be reset to 0, got %d", e)
	}

	// Verify prometheus metrics
	bytesVal := testutil.ToFloat64(ArrowBytesTransferred)
	if bytesVal == 0 {
		t.Errorf("Expected Prometheus bytes metric to be > 0, got 0")
	}

	embedVal := testutil.ToFloat64(ArrowEmbeddingsPushed)
	if embedVal == 0 {
		t.Errorf("Expected Prometheus embeddings metric to be > 0, got 0")
	}
}

// Fuzz test to ensure no race conditions during concurrent hotpath updates
func FuzzBgFlusher(f *testing.F) {
	f.Add(uint(100), uint(1000))

	f.Fuzz(func(t *testing.T, numThreads uint, iterations uint) {
		if numThreads > 1000 {
			numThreads = 1000
		}
		if iterations > 10000 {
			iterations = 10000
		}
		if numThreads == 0 || iterations == 0 {
			return
		}

		arrowBytesHotpath.Store(0)
		
		flusher := NewBgFlusher(5 * time.Millisecond)
		var wg sync.WaitGroup

		for i := uint(0); i < numThreads; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := uint(0); j < iterations; j++ {
					RecordArrowBytesHotpath(10)
					runtime.Gosched() // Yield to allow background flusher to run intermittently
				}
			}()
		}

		wg.Wait()
		time.Sleep(20 * time.Millisecond) // Allow final flush
		flusher.Stop()

		// The atomic counter should eventually go back to 0
		if finalBytes := arrowBytesHotpath.Load(); finalBytes != 0 {
			t.Errorf("Expected final atomic count to be flushed to 0, got %d", finalBytes)
		}
	})
}
