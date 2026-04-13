package metrics

import (
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// Arrow Metrics
	ArrowBytesTransferred = promauto.NewCounter(prometheus.CounterOpts{
		Name: "arrow_flight_bytes_transferred_total",
		Help: "Total bytes transferred via Arrow Flight",
	})

	ArrowEmbeddingsPushed = promauto.NewCounter(prometheus.CounterOpts{
		Name: "arrow_flight_embeddings_pushed_total",
		Help: "Total number of embeddings pushed via Arrow Flight",
	})

	BatchQueueDepth = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "batch_queue_depth",
		Help: "Current number of requests waiting for inference",
	})

	BatchRunningSequences = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "batch_running_sequences",
		Help: "Current number of active sequences being processed",
	})
)

// Atomics for hotpath updates
var (
	arrowBytesHotpath      atomic.Int64
	arrowEmbeddingsHotpath atomic.Int64
	batchQueueHotpath      atomic.Int64
	batchRunningHotpath    atomic.Int64
)

// RecordArrowBytesHotpath sets the metric without any lock contention
func RecordArrowBytesHotpath(bytes int64) {
	arrowBytesHotpath.Add(bytes)
}

// RecordArrowEmbeddingHotpath avoids locking inside generation loop
func RecordArrowEmbeddingHotpath() {
	arrowEmbeddingsHotpath.Add(1)
}

func RecordBatchStats(waiting, running, prefill int) {
	batchQueueHotpath.Store(int64(waiting))
	batchRunningHotpath.Store(int64(running + prefill))
}

// BgFlusher periodically moves atomic values into prometheus metrics.
type BgFlusher struct {
	ticker *time.Ticker
	quit   chan struct{}
}

// NewBgFlusher initializes a background flusher that updates prometheus metrics every interval.
func NewBgFlusher(interval time.Duration) *BgFlusher {
	f := &BgFlusher{
		ticker: time.NewTicker(interval),
		quit:   make(chan struct{}),
	}
	go f.run()
	return f
}

func (f *BgFlusher) run() {
	for {
		select {
		case <-f.ticker.C:
			// Flush Arrow bytes
			if b := arrowBytesHotpath.Swap(0); b > 0 {
				ArrowBytesTransferred.Add(float64(b))
			}
			// Flush embeddings
			if e := arrowEmbeddingsHotpath.Swap(0); e > 0 {
				ArrowEmbeddingsPushed.Add(float64(e))
			}
			// Flush Batch Stats
			BatchQueueDepth.Set(float64(batchQueueHotpath.Load()))
			BatchRunningSequences.Set(float64(batchRunningHotpath.Load()))
		case <-f.quit:
			f.ticker.Stop()
			return
		}
	}
}

// Stop terminates the background flusher thread.
func (f *BgFlusher) Stop() {
	close(f.quit)
}
