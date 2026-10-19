package engine

import (
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
