package engine

import (
	"math"
	"strings"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

// The audit functions in audit.go are pure diagnostics. They were
// Metal-only, uncalled, and (except for the logit range audit, which is
// now wired into the engines) completely untested.

// TestAuditLogitRange covers the logit distribution audit.
func TestAuditLogitRange(t *testing.T) {
	// An empty input is not an error, it is an empty audit.
	empty := AuditLogitRange(nil)
	if empty.HasNaN || empty.HasInf || empty.IsFlat {
		t.Errorf("an empty logit vector was reported as broken: %+v", empty)
	}

	// A healthy distribution.
	healthy := AuditLogitRange([]float32{1, 2, 3, 4})
	if healthy.Max != 4 || healthy.Min != 1 {
		t.Errorf("the range is %v..%v, want 1..4", healthy.Min, healthy.Max)
	}
	if healthy.Mean != 2.5 {
		t.Errorf("the mean is %v, want 2.5", healthy.Mean)
	}
	if healthy.RMS <= 0 {
		t.Errorf("the RMS is %v, want a positive value", healthy.RMS)
	}
	if healthy.HasNaN || healthy.HasInf || healthy.IsFlat {
		t.Errorf("a healthy distribution was reported as broken: %+v", healthy)
	}

	// A NaN.
	withNaN := AuditLogitRange([]float32{1, float32(math.NaN()), 3})
	if !withNaN.HasNaN || withNaN.NumNaNs != 1 {
		t.Errorf("the NaN was not detected: %+v", withNaN)
	}
	if !withNaN.HasExtremeValues {
		t.Error("a NaN must also count as an extreme value")
	}

	// An infinity.
	withInf := AuditLogitRange([]float32{1, float32(math.Inf(1)), 3})
	if !withInf.HasInf || withInf.NumInfs != 1 {
		t.Errorf("the infinity was not detected: %+v", withInf)
	}

	// A flat distribution: every logit the same.
	flat := AuditLogitRange([]float32{0.5, 0.5, 0.5, 0.5})
	if !flat.IsFlat {
		t.Errorf("a flat distribution was not detected: %+v", flat)
	}
}

// TestAuditKVCachePositions covers the KV cache position audit.
func TestAuditKVCachePositions(t *testing.T) {
	// Positions inside the cache, all distinct.
	ok := AuditKVCachePositions(16, []int{0, 1, 2, 3})
	if ok.HasOutOfBounds {
		t.Error("in-bounds positions were reported as out of bounds")
	}
	if ok.HasOverlap {
		t.Error("distinct positions were reported as overlapping")
	}
	if ok.UniquePositions != 4 || ok.TotalPositions != 4 {
		t.Errorf("the position counts are %d unique of %d, want 4 of 4", ok.UniquePositions, ok.TotalPositions)
	}

	// A position outside the cache.
	oob := AuditKVCachePositions(16, []int{0, 16, -1})
	if !oob.HasOutOfBounds {
		t.Error("an out-of-bounds position was not detected")
	}
	if len(oob.OutOfBoundsIndices) != 2 {
		t.Errorf("%d out-of-bounds indices were reported, want 2", len(oob.OutOfBoundsIndices))
	}

	// A repeated position: the sliding window case.
	overlap := AuditKVCachePositions(4, []int{0, 1, 0, 1})
	if !overlap.HasOverlap {
		t.Error("a repeated position was not detected")
	}
	if !overlap.IsSlidingWindow {
		t.Error("a repeated position was not reported as a sliding window")
	}
}

// TestAuditScratchBufferSizing covers the scratch buffer sizing audit.
func TestAuditScratchBufferSizing(t *testing.T) {
	audit := AuditScratchBufferSizing(1, 4096, 32, 8, 128, 2048, 32000)
	if audit.ExpectedScores != 32*2048 {
		t.Errorf("the expected score count is %d, want %d", audit.ExpectedScores, 32*2048)
	}
	if audit.ActualScores < audit.ExpectedScores {
		t.Errorf("the aligned score count %d is smaller than the expected %d", audit.ActualScores, audit.ExpectedScores)
	}
	if audit.ActualScoresSize%4096 != 0 {
		t.Errorf("the score buffer size %d is not a multiple of 4096", audit.ActualScoresSize)
	}
	if !audit.Alignment4096 {
		t.Error("the alignment flag was not set")
	}
	if audit.GQARatio != 4 {
		t.Errorf("the GQA ratio is %d, want 4", audit.GQARatio)
	}
	if !audit.IsValid {
		t.Errorf("a consistent configuration was reported as invalid: %+v", audit)
	}

	// A head dimension that does not divide the model dimension is
	// inconsistent.
	bad := AuditScratchBufferSizing(1, 100, 32, 8, 128, 2048, 32000)
	if bad.IsValid {
		t.Error("an inconsistent head dimension was reported as valid")
	}
}

// TestAuditDequantizationAccuracy covers the dequantization audit's
// contract. It is a stub: it checks the type, not the values, and its
// comment says so.
func TestAuditDequantizationAccuracy(t *testing.T) {
	// No data is a failure.
	empty := AuditDequantizationAccuracy(device.DataTypeQ4K, nil, 0)
	if empty.Passed {
		t.Error("an empty input was reported as passing")
	}
	if empty.MaxRelError != 1.0 {
		t.Errorf("the relative error is %v, want 1.0", empty.MaxRelError)
	}

	// A known type passes.
	known := AuditDequantizationAccuracy(device.DataTypeQ4K, []byte{1, 2, 3, 4}, 4)
	if !known.Passed {
		t.Error("a known quantization type was reported as failing")
	}
	if known.TotalSamples != 4 || known.NumElements != 4 {
		t.Errorf("the sample counts are %d of %d, want 4 of 4", known.TotalSamples, known.NumElements)
	}

	// An unknown type fails.
	unknown := AuditDequantizationAccuracy(device.DataTypeTQ1_0, []byte{1, 2, 3, 4}, 4)
	if unknown.Passed {
		t.Error("an unknown quantization type was reported as passing")
	}
}

// TestAuditWeightPadding covers the weight alignment audit.
func TestAuditWeightPadding(t *testing.T) {
	// A Q4_K tensor with a padded row count.
	padded := AuditWeightPadding("token_embd.weight", 100, 4096, 128, 4096, 12)
	if !padded.HasPadding {
		t.Error("the padding was not detected")
	}
	if padded.PaddingBytes != 128*4096-100*4096 {
		t.Errorf("the padding is %d bytes, want %d", padded.PaddingBytes, 128*4096-100*4096)
	}
	if padded.AlignmentReq != 256 {
		t.Errorf("the alignment requirement is %d, want 256", padded.AlignmentReq)
	}
	// The alignment requirement is on the row length: 4096 columns is a
	// whole number of 256-element Q4_K blocks, so 128 rows is fine.
	if !padded.IsAligned {
		t.Error("a Q4_K tensor with 4096 columns was reported as misaligned")
	}
	if !padded.Valid {
		t.Error("a valid tensor was reported as invalid")
	}

	// A tensor with no padding.
	exact := AuditWeightPadding("token_embd.weight", 128, 4096, 128, 4096, 12)
	if exact.HasPadding {
		t.Error("a tensor with no padding was reported as padded")
	}
	if exact.PaddingBytes != 0 {
		t.Errorf("the padding is %d bytes, want 0", exact.PaddingBytes)
	}

	// A row length that is not a whole number of blocks.
	ragged := AuditWeightPadding("token_embd.weight", 100, 4000, 100, 4000, 12)
	if ragged.IsAligned {
		t.Error("a Q4_K tensor with 4000 columns was reported as aligned")
	}

	// An F16 tensor only needs an even number of elements per row.
	f16 := AuditWeightPadding("token_embd.weight", 100, 4000, 100, 4000, 1)
	if !f16.IsAligned {
		t.Error("an F16 tensor with 4000 columns was reported as misaligned")
	}
}

// TestAuditSoftmaxMasking covers the masking audit, including the
// strictness detection it exists for.
//
// The strictness flag used to be set to false inside the loop and then
// unconditionally set to true afterwards, so the audit always reported
// strict masking -- the one thing it exists to question.
func TestAuditSoftmaxMasking(t *testing.T) {
	const seqLen = 8
	const pos = 4

	// Full attention: positions after pos are masked, the rest are not.
	scores := make([][]float32, 1)
	scores[0] = make([]float32, seqLen)
	for i := range scores[0] {
		scores[0][i] = 1
	}
	for i := pos + 1; i < seqLen; i++ {
		scores[0][i] = -100000
	}

	strict := AuditSoftmaxMasking(seqLen, pos, 0, scores)
	if !strict.IsStrict {
		t.Errorf("a correctly masked distribution was reported as not strict: %+v", strict)
	}
	if strict.NumMasked != seqLen-pos-1 {
		t.Errorf("%d positions were counted as masked, want %d", strict.NumMasked, seqLen-pos-1)
	}
	if strict.NumUnmasked != pos+1 {
		t.Errorf("%d positions were counted as unmasked, want %d", strict.NumUnmasked, pos+1)
	}
	if !strict.AllPositionsInBounds {
		t.Error("the positions were reported as out of bounds")
	}

	// A position that should be masked but is not.
	leaky := AuditSoftmaxMasking(seqLen, pos, 0, scores)
	leakyScores := make([][]float32, 1)
	leakyScores[0] = make([]float32, seqLen)
	for i := range leakyScores[0] {
		leakyScores[0][i] = 1
	}
	leaky = AuditSoftmaxMasking(seqLen, pos, 0, leakyScores)
	if leaky.IsStrict {
		t.Error("an unmasked future position was reported as strict masking")
	}

	// A position that should not be masked but is.
	overMasked := make([][]float32, 1)
	overMasked[0] = make([]float32, seqLen)
	for i := range overMasked[0] {
		overMasked[0][i] = -100000
	}
	res := AuditSoftmaxMasking(seqLen, pos, 0, overMasked)
	if res.IsStrict {
		t.Error("an incorrectly masked position was reported as strict masking")
	}

	// No scores at all.
	none := AuditSoftmaxMasking(seqLen, pos, 0, nil)
	if none.IsStrict {
		t.Error("an empty score matrix was reported as strict masking")
	}
}

// TestAuditHeadDimensionLogic covers the head dimension audit.
func TestAuditHeadDimensionLogic(t *testing.T) {
	audit := AuditHeadDimensionLogic(128, 32, 2048)
	if !audit.IsPowerOf2 {
		t.Error("128 was not reported as a power of two")
	}
	if !audit.OptimalThreadgroup {
		t.Error("a power-of-two head dimension was reported as suboptimal")
	}
	if audit.TotalThreads != 32*2048 {
		t.Errorf("the thread count is %d, want %d", audit.TotalThreads, 32*2048)
	}

	odd := AuditHeadDimensionLogic(100, 32, 2048)
	if odd.IsPowerOf2 {
		t.Error("100 was reported as a power of two")
	}
	if odd.OptimalThreadgroup {
		t.Error("a non-power-of-two head dimension was reported as optimal")
	}
}

// TestAuditActivationFlow covers the layer-by-layer activation flow audit.
func TestAuditActivationFlow(t *testing.T) {
	healthy := AuditActivationFlow([]device.ActivationStats{
		{RMS: 1, Max: 4},
		{RMS: 2, Max: 8},
		{RMS: 1.5, Max: 6},
	}, 3)
	if !healthy.IsHealthy {
		t.Errorf("a healthy flow was reported as unhealthy: %+v", healthy)
	}
	if healthy.HasJumps {
		t.Error("a smooth flow was reported as having jumps")
	}

	// A collapsed layer.
	collapsed := AuditActivationFlow([]device.ActivationStats{
		{RMS: 1, Max: 4},
		{RMS: 0, Max: 0},
	}, 2)
	if collapsed.IsHealthy {
		t.Error("a collapsed layer was not detected")
	}
	if len(collapsed.CollapsedLayers) != 1 || collapsed.CollapsedLayers[0] != 1 {
		t.Errorf("the collapsed layers are %v, want [1]", collapsed.CollapsedLayers)
	}

	// A saturated layer.
	saturated := AuditActivationFlow([]device.ActivationStats{
		{RMS: 1, Max: 4},
		{RMS: SaturationThreshold * 2, Max: SaturationThreshold * 2},
	}, 2)
	if saturated.IsHealthy {
		t.Error("a saturated layer was not detected")
	}
	if len(saturated.SaturatedLayers) != 1 {
		t.Errorf("the saturated layers are %v, want one", saturated.SaturatedLayers)
	}

	// A jump between layers.
	jumpy := AuditActivationFlow([]device.ActivationStats{
		{RMS: 1, Max: 1},
		{RMS: 1, Max: 100},
	}, 2)
	if !jumpy.HasJumps {
		t.Error("a jump between layers was not detected")
	}
	if len(jumpy.JumpDetails) != 1 {
		t.Errorf("%d jump details were reported, want 1", len(jumpy.JumpDetails))
	}
}

// TestAuditNaNPropagation covers the NaN propagation audit.
func TestAuditNaNPropagation(t *testing.T) {
	clean := AuditNaNPropagation([]device.ActivationStats{{RMS: 1, Max: 2}})
	if clean.HasNaN {
		t.Error("a clean trace was reported as having NaNs")
	}
	if clean.NaNLayerStart != -1 || clean.NaNLayerEnd != -1 {
		t.Errorf("the NaN layer range is %d..%d, want -1..-1", clean.NaNLayerStart, clean.NaNLayerEnd)
	}

	// A NaN in the last layer only: sudden.
	sudden := AuditNaNPropagation([]device.ActivationStats{
		{RMS: 1, Max: 2},
		{NaNs: 3},
	})
	if !sudden.HasNaN {
		t.Error("the NaN was not detected")
	}
	if sudden.Pattern != "sudden" {
		t.Errorf("the pattern is %q, want %q", sudden.Pattern, "sudden")
	}
	if sudden.TotalNaNCount != 3 {
		t.Errorf("the NaN count is %d, want 3", sudden.TotalNaNCount)
	}
	if sudden.NaNLayerStart != 1 || sudden.NaNLayerEnd != 1 {
		t.Errorf("the NaN layer range is %d..%d, want 1..1", sudden.NaNLayerStart, sudden.NaNLayerEnd)
	}

	// A NaN that starts early and continues: gradual.
	gradual := AuditNaNPropagation([]device.ActivationStats{
		{RMS: 1, Max: 2},
		{NaNs: 1},
		{NaNs: 1},
	})
	if gradual.Pattern != "gradual" {
		t.Errorf("the pattern is %q, want %q", gradual.Pattern, "gradual")
	}
	if gradual.NaNLayerStart != 1 || gradual.NaNLayerEnd != 2 {
		t.Errorf("the NaN layer range is %d..%d, want 1..2", gradual.NaNLayerStart, gradual.NaNLayerEnd)
	}
}

// TestAuditRoPEDeviation covers the RoPE precision audit.
func TestAuditRoPEDeviation(t *testing.T) {
	within := AuditRoPEDeviation(10000, 0.01, 0.005)
	if !within.Passed {
		t.Error("a deviation within the bound was reported as a failure")
	}
	if within.DeviationRatio != 0.5 {
		t.Errorf("the deviation ratio is %v, want 0.5", within.DeviationRatio)
	}

	beyond := AuditRoPEDeviation(10000, 0.01, 0.02)
	if beyond.Passed {
		t.Error("a deviation beyond the bound was reported as a pass")
	}
	if beyond.DeviationRatio != 2 {
		t.Errorf("the deviation ratio is %v, want 2", beyond.DeviationRatio)
	}
}

// TestAuditResultStrings covers the String methods, which the audit
// results use when they are logged.
func TestAuditResultStrings(t *testing.T) {
	logit := LogitRangeAuditResult{Max: 1, Min: -1, Mean: 0, RMS: 0.5, HasNaN: true}
	if s := logit.String(); !strings.Contains(s, "hasNaN=true") {
		t.Errorf("the logit range string is %q, want it to mention the NaN", s)
	}

	kv := KVCacheAuditResult{CacheSize: 16, HasOverlap: true, HasOutOfBounds: false, UniquePositions: 3, TotalPositions: 4}
	if s := kv.String(); !strings.Contains(s, "hasOverlap=true") {
		t.Errorf("the KV cache string is %q, want it to mention the overlap", s)
	}

	buf := BufferSizingAuditResult{ExpectedScores: 10, ActualScores: 20, IsValid: true, Alignment4096: true, GQARatio: 4}
	if s := buf.String(); !strings.Contains(s, "gqa=4") {
		t.Errorf("the buffer sizing string is %q, want it to mention the GQA ratio", s)
	}
}
