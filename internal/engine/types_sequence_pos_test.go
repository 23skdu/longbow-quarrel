package engine

import (
	"sync"
	"testing"
)

// The Metal engine's GetSeqCachePos and IncSeqCachePos used to lock a
// mutex on the Sequence struct itself, which did not exist: the Metal
// backend did not compile. The locking now lives on the SequenceManager,
// which is where the map that owns the sequences is already guarded.
//
// These tests assert the two properties the locking exists for: the
// position is read and advanced under the manager's lock, and the two
// operations do not race with each other.

// TestSequenceManager_SequencePos covers the read and the increment of a
// sequence's cached position.
func TestSequenceManager_SequencePos(t *testing.T) {
	sm := NewSequenceManager()
	sm.sequences[7] = &Sequence{ID: 7}

	pos, ok := sm.GetSequencePos(7)
	if !ok {
		t.Fatal("GetSequencePos(7) reported the sequence as missing")
	}
	if pos != 0 {
		t.Errorf("a fresh sequence has position %d, want 0", pos)
	}

	sm.IncSequencePos(7)
	sm.IncSequencePos(7)

	pos, ok = sm.GetSequencePos(7)
	if !ok {
		t.Fatal("GetSequencePos(7) reported the sequence as missing after the increments")
	}
	if pos != 2 {
		t.Errorf("after two increments the position is %d, want 2", pos)
	}
}

// TestSequenceManager_SequencePosMissing covers the two operations on a
// sequence that does not exist: neither may panic, and the read must
// report the miss.
func TestSequenceManager_SequencePosMissing(t *testing.T) {
	sm := NewSequenceManager()

	pos, ok := sm.GetSequencePos(99)
	if ok {
		t.Error("GetSequencePos reported a missing sequence as present")
	}
	if pos != 0 {
		t.Errorf("a missing sequence reported position %d, want 0", pos)
	}

	// Must not panic and must not create the sequence.
	sm.IncSequencePos(99)
	if _, present := sm.GetSequencePos(99); present {
		t.Error("IncSequencePos created the sequence it was asked to advance")
	}
}

// TestSequenceManager_SequencePosConcurrent checks that the position can
// be advanced and read at the same time without a data race. Run under
// -race, this fails if the lock is missing.
func TestSequenceManager_SequencePosConcurrent(t *testing.T) {
	sm := NewSequenceManager()
	sm.sequences[1] = &Sequence{ID: 1}

	const workers = 8
	const iterations = 200

	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				sm.IncSequencePos(1)
			}
		}()
	}
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				if _, ok := sm.GetSequencePos(1); !ok {
					t.Error("the sequence disappeared while it was being advanced")
					return
				}
			}
		}()
	}
	wg.Wait()

	pos, ok := sm.GetSequencePos(1)
	if !ok {
		t.Fatal("the sequence disappeared")
	}
	if want := workers * iterations; pos != want {
		t.Errorf("the position is %d, want %d", pos, want)
	}
}
