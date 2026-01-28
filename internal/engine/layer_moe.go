//go:build darwin && metal

package engine

import (
	"time"

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

// MOELayerForward performs MOE layer forward pass
// input: [batch_size, dim] hidden states
// Returns: [batch_size, dim] output hidden states
func (e *Engine) MOELayerForward(layerIdx int, input *device.Tensor) *device.Tensor {
	moeStart := time.Now()
	defer func() {
		metrics.RecordMOELayerLatency(time.Since(moeStart))
	}()
	moeWeights := e.Weights.MOE[layerIdx]
	if moeWeights == nil || moeWeights.Router == nil || moeWeights.Experts == nil {
		logger.Log.Warn("MOE layer has no weights, skipping", "layer", layerIdx)
		return input
	}

	batchSize := input.Rows()
	dim := input.Cols()
	numExperts := moeWeights.Experts.NumExperts
	hiddenDim := moeWeights.Experts.HiddenDim
	topK := e.Config.ExpertUsedCount

	logger.Log.Debug("MOE layer forward", "layer", layerIdx, "batch", batchSize, "dim", dim,
		"num_experts", numExperts, "hidden_dim", hiddenDim, "top_k", topK)

	// Step 1: Compute routing logits
	startTime := time.Now()
	logits := e.Ctx.MOERouterLogits(input, moeWeights.Router.GateInput)

	// Step 2: Select top-k experts
	expertIndices, expertWeights := e.Ctx.MOETopKSelection(logits, topK)
	logits.Free()
	routingDuration := time.Since(startTime)
	metrics.RecordMOERoutingLatency(routingDuration)

	// Record expert selections
	indices := expertIndices.ToHost()
	// Convert flat float32 slice to int32 slice
	intIndices := make([]int32, len(indices))
	for i, v := range indices {
		intIndices[i] = int32(v)
	}
	metrics.RecordMOEExpertSelection(layerIdx, intIndices)

	// Step 3 & 4: Compute expert outputs (fused Gate + Up + SwiGLU)
	gemmStart := time.Now()
	var activated *device.Tensor
	if moeWeights.Experts.FfnGateExperts != nil && moeWeights.Experts.FfnUpExperts != nil {
		activated = e.Ctx.MOEExpertGateUpSwiGLU(input, moeWeights.Experts.FfnGateExperts,
			moeWeights.Experts.FfnUpExperts, expertIndices, expertWeights, hiddenDim)
	} else {
		logger.Log.Warn("MOE layer missing gate or up weights", "layer", layerIdx)
		expertIndices.Free()
		expertWeights.Free()
		return input
	}

	// Step 5: Down projection
	var output *device.Tensor
	if moeWeights.Experts.FfnDownExperts != nil {
		output = e.Ctx.MOEExpertForward(activated, moeWeights.Experts.FfnDownExperts,
			expertIndices, expertWeights, dim)
		activated.Free()
	} else {
		logger.Log.Warn("MOE layer missing down weights", "layer", layerIdx)
		activated.Free()
		expertIndices.Free()
		expertWeights.Free()
		return input
	}
	gemmDuration := time.Since(gemmStart)

	logger.Log.Info("MOE breakdown", "layer", layerIdx, "routing", routingDuration, "gemm", gemmDuration)

	// Step 5: Add shared expert if present
	if moeWeights.Shared != nil && moeWeights.Shared.FfnGateShared != nil &&
		moeWeights.Shared.FfnUpShared != nil && moeWeights.Shared.FfnDownShared != nil {
		// Shared expert: standard FFN
		sharedGate, _ := input.Linear(moeWeights.Shared.FfnGateShared)
		sharedUp, _ := input.Linear(moeWeights.Shared.FfnUpShared)
		sharedActivated, _ := sharedUp.SwiGLU(sharedGate)
		sharedGate.Free()
		sharedUp.Free()

		sharedOut, _ := sharedActivated.Linear(moeWeights.Shared.FfnDownShared)
		sharedActivated.Free()

		// Add shared expert output to MOE output
		finalOutput, _ := output.Add(sharedOut)
		output.Free()
		sharedOut.Free()
		output = finalOutput
	}

	expertIndices.Free()
	expertWeights.Free()

	return output
}

// IsMOELayer checks if a layer index corresponds to an MOE layer
func (e *Engine) IsMOELayer(l int) bool {
	if l < 0 || l >= len(e.Weights.MOE) {
		return false
	}
	res := e.Weights.MOE[l] != nil
	if res {
		logger.Log.Debug("IsMOELayer true", "layer", l)
	}
	return res
}
