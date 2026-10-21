package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

// TestTheMemoryGovernorOffloadsTheColdestSequence checks that the memory
// governor is bound to the cache and that its trigger does something.
//
// The governor was triggered by the CUDA decode loop and had no handler
// registered in production, so the trigger did nothing: the cache never
// compacted and the coldest sequence was never offloaded.
func TestTheMemoryGovernorOffloadsTheColdestSequence(t *testing.T) {
	cache, ctx := newTestPagedCache(t, 512)
	defer cache.Free()
	defer ctx.Free()

	// A sequence with a block in the cache.
	if err := cache.Allocate("cold-seq", 32); err != nil {
		t.Fatalf("failed to allocate the sequence: %v", err)
	}
	if len(cache.BlockTable("cold-seq")) == 0 {
		t.Fatal("the sequence has no blocks")
	}

	// The governor is a process-wide singleton, so the samplers it uses
	// have to be put back.
	governor := device.GetMemoryGovernor()
	governor.SetCustomMemorySamplers(
		func() (used, total int64, pct float64) { return 95, 100, 0.99 },
		func() (used, total int64, pct float64) { return 95, 100, 0.99 },
	)
	defer governor.SetCustomMemorySamplers(nil, nil)

	cache.RegisterMemoryGovernor(governor)

	action := governor.TriggerGovernor(0.5)
	if action == "none" {
		t.Fatal("the governor did nothing under 99 % memory pressure")
	}

	if !cache.IsOffloaded("cold-seq") {
		t.Error("the governor did not offload the only sequence")
	}

	// The registration is remembered: the governor is a singleton, so a
	// second registration would make every handler fire twice.
	cache.RegisterMemoryGovernor(governor)
}

// TestTheTouchedSequenceIsNotTheColdest checks that the cache records
// which sequence was used.
//
// The timestamp used to be written only when a sequence was restored from
// the host, so the "least recently used" sequence was whichever one the
// map iteration happened to yield first.
func TestTheTouchedSequenceIsNotTheColdest(t *testing.T) {
	cache, ctx := newTestPagedCache(t, 512)
	defer cache.Free()
	defer ctx.Free()

	if err := cache.Allocate("seq-a", 32); err != nil {
		t.Fatalf("failed to allocate the sequence: %v", err)
	}
	if err := cache.Allocate("seq-b", 32); err != nil {
		t.Fatalf("failed to allocate the sequence: %v", err)
	}

	cache.Touch("seq-b")

	seqID, err := cache.OffloadLRU()
	if err != nil {
		t.Fatalf("the offload failed: %v", err)
	}
	if seqID != "seq-a" {
		t.Errorf("the offloaded sequence is %q, want the one that was not touched (%q)", seqID, "seq-a")
	}
}
