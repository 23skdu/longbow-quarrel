package engine

import (
	"fmt"
	"sync"

	"testing"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

// TestSequenceManager_GetSequence covers lookup semantics of the sequence
// registry: a miss must report found=false with a nil pointer, and a hit
// must return the same instance that was registered.
func TestSequenceManager_GetSequence(t *testing.T) {
	sm := NewSequenceManager()

	if sm == nil {
		t.Fatal("NewSequenceManager returned nil")
	}

	if seq, ok := sm.GetSequence(1); ok {
		t.Errorf("GetSequence on an empty manager reported found=true (seq=%v)", seq)
	} else if seq != nil {
		t.Errorf("GetSequence miss returned a non-nil sequence: %v", seq)
	}

	want := &Sequence{ID: 7, MaxTokens: 128}
	sm.sequences[7] = want

	got, ok := sm.GetSequence(7)
	if !ok {
		t.Fatal("GetSequence(7) reported found=false after registration")
	}
	if got != want {
		t.Errorf("GetSequence(7) returned %p, want the registered instance %p", got, want)
	}
	if got.MaxTokens != 128 {
		t.Errorf("MaxTokens = %d, want 128", got.MaxTokens)
	}
}

// TestSequenceManager_ConcurrentGetSequence exercises the RWMutex under
// -race: concurrent readers must not race with each other or with a writer,
// and every read must observe a fully constructed sequence. Failures observed
// inside the goroutines are collected and asserted in the test body, because
// calling t.Errorf off the test goroutine is not permitted.
func TestSequenceManager_ConcurrentGetSequence(t *testing.T) {
	sm := NewSequenceManager()
	sm.sequences[1] = &Sequence{ID: 1, MaxTokens: 64}

	const readers = 8
	const iterations = 200

	var (
		mu       sync.Mutex
		failures []string
	)
	record := func(format string, args ...any) {
		mu.Lock()
		defer mu.Unlock()
		failures = append(failures, fmt.Sprintf(format, args...))
	}

	var wg sync.WaitGroup
	for r := 0; r < readers; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				seq, ok := sm.GetSequence(1)
				if !ok {
					record("sequence 1 disappeared during concurrent reads")
					return
				}
				if seq.MaxTokens != 64 {
					record("MaxTokens = %d, want 64", seq.MaxTokens)
					return
				}
			}
		}()
	}

	// A writer taking the write lock forces real RLock contention.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			sm.mu.Lock()
			sm.sequences[1] = &Sequence{ID: 1, MaxTokens: 64}
			sm.mu.Unlock()
		}
	}()

	wg.Wait()

	if len(failures) != 0 {
		t.Fatalf("concurrent access produced %d failure(s): %v", len(failures), failures)
	}

	// The registry must still be intact and readable once the storm settles.
	seq, ok := sm.GetSequence(1)
	if !ok {
		t.Fatal("sequence 1 missing after concurrent readers finished")
	}
	if seq.ID != 1 {
		t.Errorf("sequence ID = %d, want 1", seq.ID)
	}
}

// TestSequenceManager_ZeroValueIsUsable checks the type still works if a
// caller embeds it without going through the constructor, since GetSequence
// reads the map under a lock and a nil map read is legal.
func TestSequenceManager_ZeroValueIsUsable(t *testing.T) {
	var sm SequenceManager

	if seq, ok := sm.GetSequence(42); ok || seq != nil {
		t.Errorf("zero-value GetSequence(42) = (%v, %v), want (nil, false)", seq, ok)
	}
}

// TestWeightsFree_NilReceiversAreSafe covers the nil-receiver guards on the
// weight cleanup helpers. These run on teardown paths that also fire during
// partial model loads, so a nil receiver must not panic. Each case also
// asserts that Free is idempotent and leaves the receiver untouched, so a
// double teardown stays safe.
func TestWeightsFree_NilReceiversAreSafe(t *testing.T) {
	t.Run("MambaWeights", func(t *testing.T) {
		var nilW *MambaWeights
		nilW.Free() // must not panic
		nilW.Free()

		w := &MambaWeights{} // every field nil
		w.Free()
		w.Free() // idempotent

		if w.A != nil || w.InWeight != nil || w.NormBias != nil {
			t.Errorf("Free mutated a MambaWeights that had only nil fields: %+v", w)
		}
	})

	t.Run("LlamaWeights", func(t *testing.T) {
		var nilW *LlamaWeights
		nilW.Free()
		nilW.Free()

		w := &LlamaWeights{}
		w.Free()
		w.Free()

		if w.TokenEmb != nil || w.Output != nil {
			t.Errorf("Free mutated an empty LlamaWeights: %+v", w)
		}
	})

	t.Run("MOEExpertWeights", func(t *testing.T) {
		var nilW *MOEExpertWeights
		nilW.Free()
		nilW.Free()

		w := &MOEExpertWeights{}
		w.Free()
		w.Free()

		if w.FfnGateExperts != nil || w.FfnDownExperts != nil {
			t.Errorf("Free mutated an empty MOEExpertWeights: %+v", w)
		}
	})

	t.Run("MOELayerWeights", func(t *testing.T) {
		var nilW *MOELayerWeights
		nilW.Free()
		nilW.Free()

		w := &MOELayerWeights{}
		w.Free()
		w.Free()
	})
}

// TestLlamaWeightsFree_NilSliceEntriesAreSkipped exercises the slice-cleanup
// helpers with nil elements mixed in, which is what a partially loaded model
// leaves behind. Free must skip them, survive being called twice, and leave
// the slice contents alone.
func TestLlamaWeightsFree_NilSliceEntriesAreSkipped(t *testing.T) {
	w := &LlamaWeights{
		AttnQ: []*device.Tensor{nil, nil},
		AttnK: []*device.Tensor{nil},
		FfnUp: []*device.Tensor{nil, nil, nil},
		Mamba: []*MambaWeights{nil, nil},
		MOE:   []*MOELayerWeights{nil},
	}

	w.Free() // must not dereference a nil element
	w.Free() // idempotent

	if len(w.AttnQ) != 2 || len(w.FfnUp) != 3 || len(w.Mamba) != 2 || len(w.MOE) != 1 {
		t.Errorf("Free mutated slice lengths: AttnQ=%d FfnUp=%d Mamba=%d MOE=%d",
			len(w.AttnQ), len(w.FfnUp), len(w.Mamba), len(w.MOE))
	}
	for i, tt := range w.AttnQ {
		if tt != nil {
			t.Errorf("AttnQ[%d] = %v, want nil to be left in place", i, tt)
		}
	}
}
