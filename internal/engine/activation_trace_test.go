//go:build darwin && metal

package engine

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestActivationTraceTracker_New(t *testing.T) {
	tracker := NewActivationTraceTracker(32)

	if tracker == nil {
		t.Fatal("NewActivationTraceTracker returned nil")
	}

	if tracker.NumLayers != 32 {
		t.Errorf("Expected NumLayers=32, got %d", tracker.NumLayers)
	}

	if len(tracker.Traces) != 0 {
		t.Errorf("Expected empty traces on init, got %d traces", len(tracker.Traces))
	}
}

func TestActivationTraceTracker_RecordingStats(t *testing.T) {
	tracker := NewActivationTraceTracker(4)

	stats := device.ActivationStats{
		Max:    10.5,
		Min:    -2.3,
		Mean:   1.2,
		RMS:    3.4,
		Zeros:  5,
		NaNs:   0,
		Infs:   0,
		Sample: []float32{1.0, 2.0, 3.0, 4.0},
	}

	tracker.RecordLayer("embedding", 0, stats)

	if len(tracker.Traces) != 1 {
		t.Errorf("Expected 1 trace after recording, got %d", len(tracker.Traces))
	}

	trace := tracker.Traces[0]
	if trace.LayerName != "embedding" {
		t.Errorf("Expected layer name 'embedding', got '%s'", trace.LayerName)
	}
	if trace.LayerIdx != 0 {
		t.Errorf("Expected layer idx 0, got %d", trace.LayerIdx)
	}
	if trace.Max != 10.5 {
		t.Errorf("Expected max 10.5, got %f", trace.Max)
	}

	stats2 := device.ActivationStats{
		Max:    20.1,
		Min:    -5.0,
		Mean:   2.5,
		RMS:    5.0,
		Zeros:  10,
		NaNs:   1,
		Infs:   0,
		Sample: []float32{5.0, 6.0, 7.0, 8.0},
	}

	tracker.RecordLayer("attn_output", 1, stats2)

	if len(tracker.Traces) != 2 {
		t.Errorf("Expected 2 traces after recording, got %d", len(tracker.Traces))
	}
}

func TestActivationTraceTracker_CollapseDetection(t *testing.T) {
	tracker := NewActivationTraceTracker(1)

	normalStats := device.ActivationStats{
		Max:  5.0,
		Min:  -5.0,
		Mean: 1.0,
		RMS:  2.5,
	}
	tracker.RecordLayer("normal", 0, normalStats)

	isCollapsed := tracker.IsLayerCollapsed(0)
	if isCollapsed {
		t.Error("Normal activations should not be detected as collapsed")
	}

	collapsedStats := device.ActivationStats{
		Max:  0.00001,
		Min:  -0.00001,
		Mean: 0.000001,
		RMS:  0.000005,
	}
	tracker.RecordLayer("collapsed", 1, collapsedStats)

	isCollapsed = tracker.IsLayerCollapsed(1)
	if !isCollapsed {
		t.Error("Very small activations should be detected as collapsed")
	}

	zeroStats := device.ActivationStats{
		Max:   0.0,
		Min:   0.0,
		Mean:  0.0,
		RMS:   0.0,
		Zeros: 100,
	}
	tracker.RecordLayer("zeros", 2, zeroStats)

	isCollapsed = tracker.IsLayerCollapsed(2)
	if !isCollapsed {
		t.Error("All zero activations should be detected as collapsed")
	}
}

func TestActivationTraceTracker_SaturationDetection(t *testing.T) {
	tracker := NewActivationTraceTracker(1)

	normalStats := device.ActivationStats{
		Max:  10.0,
		Min:  -10.0,
		Mean: 1.0,
		RMS:  5.0,
	}
	tracker.RecordLayer("normal", 0, normalStats)

	isSaturated := tracker.IsLayerSaturated(0)
	if isSaturated {
		t.Error("Normal activations should not be detected as saturated")
	}

	largeStats := device.ActivationStats{
		Max:  100000.0,
		Min:  -100000.0,
		Mean: 5000.0,
		RMS:  10000.0,
	}
	tracker.RecordLayer("large", 1, largeStats)

	isSaturated = tracker.IsLayerSaturated(1)
	if !isSaturated {
		t.Error("Very large activations should be detected as saturated")
	}

	infStats := device.ActivationStats{
		Max:  float32(math.Inf(1)),
		Min:  -100.0,
		Mean: 1000.0,
		RMS:  float32(math.Inf(1)),
		Infs: 5,
	}
	tracker.RecordLayer("infinity", 2, infStats)

	isSaturated = tracker.IsLayerSaturated(2)
	if !isSaturated {
		t.Error("Activations with infinity should be detected as saturated")
	}
}

func TestActivationTraceTracker_GetCollapsedLayers(t *testing.T) {
	tracker := NewActivationTraceTracker(5)

	for i := 0; i < 5; i++ {
		stats := device.ActivationStats{
			Max:  float32(10.0),
			Min:  -10.0,
			Mean: 1.0,
			RMS:  5.0,
		}
		if i == 1 || i == 3 {
			stats.Max = 0.00001
			stats.Min = -0.00001
			stats.Mean = 0.000001
			stats.RMS = 0.000005
		}
		tracker.RecordLayer("test", i, stats)
	}

	collapsed := tracker.GetCollapsedLayers()
	if len(collapsed) != 2 {
		t.Errorf("Expected 2 collapsed layers, got %d", len(collapsed))
	}

	if collapsed[0] != 1 && collapsed[0] != 3 {
		t.Errorf("Expected collapsed layers to be 1 or 3, got %v", collapsed)
	}
}

func TestActivationTraceTracker_GetSaturatedLayers(t *testing.T) {
	tracker := NewActivationTraceTracker(5)

	for i := 0; i < 5; i++ {
		stats := device.ActivationStats{
			Max:  10.0,
			Min:  -10.0,
			Mean: 1.0,
			RMS:  5.0,
		}
		if i == 2 || i == 4 {
			stats.Max = 100000.0
			stats.Min = -100000.0
			stats.Mean = 5000.0
			stats.RMS = 10000.0
		}
		tracker.RecordLayer("test", i, stats)
	}

	saturated := tracker.GetSaturatedLayers()
	if len(saturated) != 2 {
		t.Errorf("Expected 2 saturated layers, got %d", len(saturated))
	}

	if saturated[0] != 2 && saturated[0] != 4 {
		t.Errorf("Expected saturated layers to be 2 or 4, got %v", saturated)
	}
}

func TestActivationTraceTracker_SaveToFile(t *testing.T) {
	tracker := NewActivationTraceTracker(3)

	stats := device.ActivationStats{
		Max:    10.0,
		Min:    -5.0,
		Mean:   1.5,
		RMS:    3.0,
		Zeros:  2,
		NaNs:   0,
		Infs:   0,
		Sample: []float32{1.0, 2.0, 3.0},
	}

	for i := 0; i < 3; i++ {
		tracker.RecordLayer("test_layer", i, stats)
	}

	tempFile := "test_trace.json"
	defer os.Remove(tempFile)

	err := tracker.SaveToFile(tempFile)
	if err != nil {
		t.Fatalf("SaveToFile failed: %v", err)
	}

	if _, err := os.Stat(tempFile); os.IsNotExist(err) {
		t.Fatal("Trace file was not created")
	}

	content, err := os.ReadFile(tempFile)
	if err != nil {
		t.Fatalf("Failed to read trace file: %v", err)
	}

	if len(content) == 0 {
		t.Error("Trace file is empty")
	}

	contentStr := string(content)
	if !strings.Contains(contentStr, "traces") {
		t.Error("Trace file missing 'traces' field")
	}
	if !strings.Contains(contentStr, "layer_name") {
		t.Error("Trace file missing 'layer_name' field")
	}
}

func TestActivationTraceTracker_ExportJSON(t *testing.T) {
	tracker := NewActivationTraceTracker(2)

	stats := device.ActivationStats{
		Max:    5.0,
		Min:    -2.0,
		Mean:   1.0,
		RMS:    2.5,
		Zeros:  1,
		NaNs:   0,
		Infs:   0,
		Sample: []float32{1.0, 2.0},
	}

	tracker.RecordLayer("layer1", 0, stats)
	tracker.RecordLayer("layer2", 1, stats)

	jsonData, err := tracker.ExportJSON()
	if err != nil {
		t.Fatalf("ExportJSON failed: %v", err)
	}

	if len(jsonData) == 0 {
		t.Error("ExportJSON returned empty data")
	}

	jsonStr := string(jsonData)
	if !strings.Contains(jsonStr, "layer1") || !strings.Contains(jsonStr, "layer2") {
		t.Error("ExportJSON missing layer data")
	}
}

func TestActivationTraceTracker_Enabled(t *testing.T) {
	tracker := NewActivationTraceTracker(10)

	if !tracker.IsEnabled() {
		t.Error("Expected tracker to be enabled by default")
	}

	tracker.Disable()
	if tracker.IsEnabled() {
		t.Error("Expected tracker to be disabled after Disable()")
	}

	tracker.Enable()
	if !tracker.IsEnabled() {
		t.Error("Expected tracker to be enabled after Enable()")
	}
}

func TestActivationTraceTracker_TrackFirstTokenOnly(t *testing.T) {
	tracker := NewActivationTraceTracker(10)

	for pos := 0; pos < 3; pos++ {
		stats := device.ActivationStats{
			Max:  float32(float32(pos) * 10.0),
			Min:  float32(float32(pos) * -10.0),
			Mean: float32(pos) + 1.0,
			RMS:  float32(pos) + 5.0,
		}

		for layer := 0; layer < 3; layer++ {
			tracker.RecordLayer("layer", layer, stats)
		}
	}

	if len(tracker.Traces) != 9 {
		t.Errorf("Expected 9 traces, got %d", len(tracker.Traces))
	}

	firstTokenTraces := tracker.GetFirstTokenTraces()
	if len(firstTokenTraces) != 3 {
		t.Errorf("Expected 3 traces for first token, got %d", len(firstTokenTraces))
	}
}
