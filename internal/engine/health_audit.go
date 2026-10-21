package engine

import (
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

// auditLogits inspects the logits a forward pass produced and records the
// result.
//
// The audit functions in audit.go and the "logit range audit" metrics both
// existed, but nothing ever called either: the metrics were permanently
// zero and a model that produced NaNs or a flat distribution said nothing.
//
// The check is O(vocab), which is negligible next to the forward pass that
// produced the logits, so it runs on every request.
func auditLogits(logits []float32) LogitRangeAuditResult {
	audit := AuditLogitRange(logits)

	metrics.RecordLogitAudit(
		audit.Max, audit.Min, audit.Mean, audit.RMS,
		audit.NumNaNs, audit.NumInfs,
		audit.HasExtremeValues, audit.IsFlat,
	)

	if audit.HasNaN || audit.HasInf {
		logger.Log.Error("the model produced non-finite logits",
			"nan", audit.NumNaNs, "inf", audit.NumInfs)
	} else if audit.IsFlat {
		logger.Log.Warn("the model produced a flat logit distribution",
			"max", audit.Max, "min", audit.Min, "flatness_ratio", audit.FlatnessRatio)
	} else if audit.HasExtremeValues {
		logger.Log.Warn("the model produced extreme logits",
			"max", audit.Max, "min", audit.Min)
	}

	return audit
}

func auditKVCache(cacheSize int, positions []int) KVCacheAuditResult {
	res := AuditKVCachePositions(cacheSize, positions)
	metrics.RecordKVCacheAudit(map[string]interface{}{
		"UniquePositions": res.UniquePositions,
		"HasOverlap":      res.HasOverlap,
		"HasOutOfBounds":  res.HasOutOfBounds,
	})
	if res.HasOutOfBounds {
		for _, pos := range res.OutOfBoundsIndices {
			metrics.RecordKVCacheOutOfBounds(pos, cacheSize)
		}
	}
	return res
}

func auditBufferSizing(batch, dim, heads, kvHeads, headDim, seqLen, vocabSize int) BufferSizingAuditResult {
	res := AuditScratchBufferSizing(batch, dim, heads, kvHeads, headDim, seqLen, vocabSize)
	metrics.RecordBufferSizingAudit(map[string]interface{}{
		"GQARatio": res.GQARatio,
	})
	return res
}

func auditDequantization(quantType device.DataType, inputData []byte, numElements int) DequantizationAuditResult {
	res := AuditDequantizationAccuracy(quantType, inputData, numElements)
	metrics.RecordDequantizationAudit(map[string]interface{}{
		"MaxAbsError": res.MaxAbsError,
		"MaxRelError": res.MaxRelError,
	})
	return res
}

func auditWeightPadding(tensorName string, reportedRows, reportedCols, actualRows, actualCols, ggufType int) WeightAlignmentAuditResult {
	res := AuditWeightPadding(tensorName, reportedRows, reportedCols, actualRows, actualCols, ggufType)
	metrics.RecordWeightAlignmentAudit(map[string]interface{}{
		"PaddingBytes": res.PaddingBytes,
	})
	return res
}

func auditSoftmaxMasking(seqLen, pos, windowSize int, scores [][]float32) SoftmaxMaskingAuditResult {
	res := AuditSoftmaxMasking(seqLen, pos, windowSize, scores)
	metrics.RecordSoftmaxMaskingAudit(map[string]interface{}{
		"MaxMaskValue": float64(res.MaskValue),
	})
	return res
}

func auditHeadDimension(headDim, numHeads, seqLen int) HeadDimensionAuditResult {
	res := AuditHeadDimensionLogic(headDim, numHeads, seqLen)
	metrics.RecordHeadDimensionAudit(map[string]interface{}{
		"ThreadgroupSize": res.ThreadgroupSize,
	})
	return res
}

func auditActivationFlow(traces []device.ActivationStats, numLayers int) ActivationFlowAuditResult {
	res := AuditActivationFlow(traces, numLayers)
	jumps := 0
	if res.HasJumps {
		jumps = 1
	}
	metrics.RecordActivationFlowAudit(len(res.CollapsedLayers), len(res.SaturatedLayers), jumps)
	return res
}

func auditNaNPropagation(traces []device.ActivationStats) NaNPropagationAuditResult {
	res := AuditNaNPropagation(traces)
	metrics.RecordNaNPropagationAudit(res.NaNLayerStart, res.NaNLayerEnd, res.TotalNaNCount, res.Pattern)
	return res
}

func auditRoPEDeviation(theta, expectedMaxDev, actualMaxDev float32) RoPEDeviationAuditResult {
	res := AuditRoPEDeviation(theta, expectedMaxDev, actualMaxDev)
	metrics.RecordRoPEDeviationAudit(res.ActualMaxDev, res.DeviationRatio, res.Passed)
	return res
}

func auditActivationPrecision(actType string, maxVal float32) {
	metrics.RecordActivationPrecision(actType, maxVal)
}
