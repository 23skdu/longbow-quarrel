//go:build webui

package engine

import (
	"context"
	"errors"
	"testing"
	"time"
)

// TestAdapterInfer_DeliversInnerChannel is the regression guard for the
// WebUI inference protocol.
//
// The caller does:
//
//	outer, err := adapter.Infer(ctx, req)
//	inner := <-outer
//	for resp := range inner { ... }
//
// so the adapter must deliver the inner channel on the outer channel. The
// CUDA and Metal adapters used to return an outer channel that nothing ever
// wrote to, and to try to *take* the inner channel back out of the outer one
// on the error path. Both meant the caller blocked forever: the WebUI's
// generate endpoint hung on every request.
//
// The CPU adapter (the only one that runs without a GPU) now follows the
// same protocol.
func TestAdapterInfer_DeliversInnerChannel(t *testing.T) {
	adapter := GetAdapter()

	req := &InferenceRequest{
		Prompt:      "hello",
		Model:       "a-model-that-does-not-exist",
		MaxTokens:   4,
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.95,
	}

	outer, err := adapter.Infer(context.Background(), req)
	if err != nil {
		// A full queue is the only acceptable error here.
		if !errors.Is(err, ErrQueueFull) {
			t.Fatalf("unexpected error from Infer: %v", err)
		}
		return
	}
	if outer == nil {
		t.Fatal("Infer returned a nil channel")
	}

	// The inner channel must arrive, otherwise the caller blocks forever.
	var inner chan InferenceResponse
	select {
	case inner = <-outer:
	case <-time.After(10 * time.Second):
		t.Fatal("the adapter never delivered the inner channel; " +
			"the caller would block forever")
	}

	if inner == nil {
		t.Fatal("the adapter delivered a nil inner channel")
	}

	// The response must be marked complete so the caller stops waiting, and
	// it must carry the error: a failure has to be distinguishable from a
	// model that produced no tokens.
	var resp InferenceResponse
	select {
	case resp = <-inner:
	case <-time.After(10 * time.Second):
		t.Fatal("the inner channel never produced a response")
	}

	if !resp.Complete {
		t.Error("expected the response to be marked complete")
	}
	if resp.Error == "" {
		t.Error("expected the response to carry an error: the model could not be loaded")
	}
}

// TestAdapterInfer_QueueFullIsAnError checks that a saturated queue is
// reported as an error rather than as a nil channel and a nil error, which a
// caller cannot distinguish from a successful submission.
func TestAdapterInfer_QueueFullIsAnError(t *testing.T) {
	adapter := GetAdapter()

	// Fill the queue. The adapter's queue holds 100 requests, and each one
	// is consumed by the background worker, so this only checks that a
	// submission either succeeds or fails explicitly -- never silently.
	for i := 0; i < 20; i++ {
		req := &InferenceRequest{
			Prompt:    "hello",
			Model:     "a-model-that-does-not-exist",
			MaxTokens: 1,
		}
		outer, err := adapter.Infer(context.Background(), req)
		if err != nil {
			if !errors.Is(err, ErrQueueFull) {
				t.Fatalf("unexpected error from Infer: %v", err)
			}
			return
		}
		if outer == nil {
			t.Fatal("Infer returned a nil channel without an error")
		}
	}
}
