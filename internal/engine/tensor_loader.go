//go:build darwin && metal

package engine

import (
	"fmt"
	"strings"

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/logger"
)

type TensorLoader struct {
	ctx     *device.Context
	config  *EngineConfig
	weights *LlamaWeights
}

func NewTensorLoader(ctx *device.Context, weights *LlamaWeights) *TensorLoader {
	return &TensorLoader{
		ctx:     ctx,
		weights: weights,
	}
}

func (tl *TensorLoader) LoadTensor(t *gguf.GGUFTensor) (*device.Tensor, error) {
	rows, cols := tl.getTensorDims(t)
	numElements := rows * cols

	var mt *device.Tensor

	switch t.Type {
	case gguf.GGMLTypeF32:
		mt = tl.loadFP32(t, numElements, rows, cols)
	case gguf.GGMLTypeF16:
		mt = tl.loadFP16(t, numElements, rows, cols, t.Name)
	case gguf.GGMLTypeQ4_K:
		mt = tl.loadQ4K(t, numElements, rows, cols)
	case gguf.GGMLTypeQ4_0:
		mt = tl.loadQ40(t, numElements, rows, cols)
	case gguf.GGMLTypeQ8_0:
		mt = tl.loadQ80(t, numElements, rows, cols)
	case gguf.GGMLTypeQ6_K:
		mt = tl.loadQ6K(t, numElements, rows, cols, t.Name)
	default:
		return nil, fmt.Errorf("unsupported tensor type: %v", t.Type)
	}

	if mt != nil && t.Type == gguf.GGMLTypeQ4_K {
		mt.ScanQ4KScales(t.Name)
	}

	return mt, nil
}

func (tl *TensorLoader) getTensorDims(t *gguf.GGUFTensor) (int, int) {
	cols := int(t.Dimensions[0])
	rows := 1
	for i := 1; i < len(t.Dimensions); i++ {
		rows *= int(t.Dimensions[i])
	}
	return rows, cols
}

func (tl *TensorLoader) loadFP32(t *gguf.GGUFTensor, numElements, rows, cols int) *device.Tensor {
	dataBytes := numElements * 4
	if uint64(len(t.Data)) < uint64(dataBytes) {
		logger.Log.Error("tensor data truncated", "name", t.Name, "expected", dataBytes, "actual", len(t.Data))
		return nil
	}

	mt := tl.ctx.NewTensorFP32(rows, cols)
	mt.LoadFrom(t.Data[:dataBytes])
	return mt
}

func (tl *TensorLoader) loadFP16(t *gguf.GGUFTensor, numElements, rows, cols int, name string) *device.Tensor {
	if isNormWeight(name) {
		f32Data := gguf.DequantizeF16(t.Data, numElements)
		mt := tl.ctx.NewTensorFP32(rows, cols)
		mt.LoadFrom(f32Data)
		return mt
	}

	mt := tl.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
	dataBytes := numElements * 2
	if uint64(len(t.Data)) < uint64(dataBytes) {
		logger.Log.Error("tensor data truncated", "name", t.Name)
		return nil
	}
	mt.LoadFromRaw(t.Data[:dataBytes])
	return mt
}

func (tl *TensorLoader) loadQ4K(t *gguf.GGUFTensor, numElements, rows, cols int) *device.Tensor {
	smallModel := tl.config != nil && tl.config.Dim < 1024

	if smallModel {
		f32Data := gguf.DequantizeQ4K(t.Data, numElements)
		mt := tl.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
		mt.LoadFrom(f32Data)
		return mt
	}

	mt, err := tl.ctx.NewQ4KTensor(rows, cols)
	if err != nil {
		logger.Log.Error("failed to create Q4K tensor", "name", t.Name, "error", err)
		return nil
	}

	dataBytes := (numElements / 256) * 144
	if uint64(len(t.Data)) < uint64(dataBytes) {
		logger.Log.Error("tensor data truncated", "name", t.Name, "expected", dataBytes, "actual", len(t.Data))
		return nil
	}
	mt.LoadFromRaw(t.Data[:dataBytes])
	return mt
}

func (tl *TensorLoader) loadQ40(t *gguf.GGUFTensor, numElements, rows, cols int) *device.Tensor {
	if numElements%32 != 0 {
		logger.Log.Error("Q4_0 tensor size not aligned", "name", t.Name, "size", numElements)
		return nil
	}

	mt := tl.ctx.NewTensorWithType(rows, cols, device.DataTypeQ4_0)
	dataBytes := (numElements / 32) * 18
	if uint64(len(t.Data)) < uint64(dataBytes) {
		logger.Log.Error("tensor data truncated", "name", t.Name)
		return nil
	}
	mt.LoadFromRaw(t.Data[:dataBytes])
	return mt
}

func (tl *TensorLoader) loadQ80(t *gguf.GGUFTensor, numElements, rows, cols int) *device.Tensor {
	f32Data := gguf.DequantizeQ8_0(t.Data, numElements)
	mt := tl.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
	mt.LoadFrom(f32Data)
	return mt
}

func (tl *TensorLoader) loadQ6K(t *gguf.GGUFTensor, numElements, rows, cols int, name string) *device.Tensor {
	largeModel := tl.config != nil && tl.config.Dim >= 1024

	if name == "output.weight" || largeModel {
		mt := tl.ctx.NewTensorWithType(rows, cols, device.DataTypeQ6K)
		dataBytes := (numElements / 256) * 210
		if uint64(len(t.Data)) < uint64(dataBytes) {
			logger.Log.Error("tensor data truncated", "name", name)
			return nil
		}
		mt.LoadFromRaw(t.Data[:dataBytes])
		return mt
	}

	f32Data := gguf.DequantizeQ6K(t.Data, numElements)
	mt := tl.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
	mt.LoadFrom(f32Data)
	return mt
}

func (tl *TensorLoader) MapTensor(mt *device.Tensor, name string, layers int) {
	lowerName := strings.ToLower(name)

	if tl.isGlobalTokenEmbedding(lowerName, name) {
		tl.weights.TokenEmb = mt
		return
	}
	if tl.isGlobalOutputNorm(name) {
		tl.weights.OutputNorm = mt
		return
	}
	if tl.isGlobalOutput(name) {
		tl.weights.Output = mt
		return
	}

	if !strings.HasPrefix(name, "blk.") {
		return
	}

	parts := strings.Split(name, ".")
	if len(parts) < 3 {
		return
	}

	layerIdx := 0
	if n, err := fmt.Sscanf(parts[1], "%d", &layerIdx); n != 1 || err != nil {
		return
	}
	if layerIdx >= layers {
		return
	}

	suffix := strings.Join(parts[2:], ".")
	tl.mapLayerWeight(mt, layerIdx, suffix)
}

func (tl *TensorLoader) isGlobalTokenEmbedding(lowerName, name string) bool {
	return (strings.HasSuffix(lowerName, "token_embd.weight") ||
		strings.HasSuffix(lowerName, "embed_tokens.weight") ||
		lowerName == "model.embed_tokens.weight") && !strings.Contains(lowerName, "blk.")
}

func (tl *TensorLoader) isGlobalOutputNorm(name string) bool {
	return (strings.HasSuffix(name, "output_norm.weight") ||
		strings.HasSuffix(name, "model.norm.weight")) && !strings.Contains(name, "blk.")
}

func (tl *TensorLoader) isGlobalOutput(name string) bool {
	return (strings.HasSuffix(name, "output.weight") ||
		name == "model.lm_head.weight") && !strings.Contains(name, "blk.")
}

func (tl *TensorLoader) mapLayerWeight(mt *device.Tensor, layerIdx int, suffix string) {
	w := tl.weights

	switch suffix {
	case "attn_q.weight":
		w.AttnQ[layerIdx] = mt
	case "attn_k.weight":
		w.AttnK[layerIdx] = mt
	case "attn_v.weight":
		w.AttnV[layerIdx] = mt
	case "attn_output.weight":
		w.AttnO[layerIdx] = mt
	case "attn_norm.weight":
		w.AttnNorm[layerIdx] = mt
	case "ffn_gate.weight":
		w.FfnGate[layerIdx] = mt
	case "ffn_down.weight":
		w.FfnDown[layerIdx] = mt
	case "ffn_up.weight":
		w.FfnUp[layerIdx] = mt
	case "ffn_norm.weight":
		w.FfnNorm[layerIdx] = mt
	case "ssm_a", "ssm_d", "ssm_conv1d.weight", "ssm_conv1d.bias",
		"ssm_dt.weight", "ssm_dt.bias", "ssm_norm.weight", "ssm_norm.bias",
		"ssm_out.weight", "ssm_in.weight":
		tl.mapMambaWeight(mt, layerIdx, suffix)
	case "ffn_gate_inp.weight", "exp_probs_b.bias",
		"ffn_down_exps.weight", "ffn_up_exps.weight", "ffn_gate_exps.weight",
		"ffn_down_shexp.weight", "ffn_up_shexp.weight", "ffn_gate_shexp.weight":
		tl.mapMOEWeight(mt, layerIdx, suffix)
	}
}

func (tl *TensorLoader) mapMambaWeight(mt *device.Tensor, layerIdx int, suffix string) {
	if tl.weights.Mamba[layerIdx] == nil {
		tl.weights.Mamba[layerIdx] = &MambaWeights{}
	}
	mw := tl.weights.Mamba[layerIdx]

	switch suffix {
	case "ssm_a":
		mw.A = mt
	case "ssm_d":
		mw.D = mt
	case "ssm_conv1d.weight":
		mw.Conv1dWeight = mt
	case "ssm_conv1d.bias":
		mw.Conv1dBias = mt
	case "ssm_dt.weight":
		mw.DTWeight = mt
	case "ssm_dt.bias":
		mw.DTBias = mt
	case "ssm_norm.weight":
		mw.NormWeight = mt
	case "ssm_norm.bias":
		mw.NormBias = mt
	case "ssm_out.weight":
		mw.OutWeight = mt
	case "ssm_in.weight":
		mw.InWeight = mt
	}
}

func (tl *TensorLoader) mapMOEWeight(mt *device.Tensor, layerIdx int, suffix string) {
	if tl.weights.MOE[layerIdx] == nil {
		tl.weights.MOE[layerIdx] = &MOELayerWeights{}
	}
	moe := tl.weights.MOE[layerIdx]

	switch suffix {
	case "ffn_gate_inp.weight":
		if moe.Router == nil {
			moe.Router = &MOERouterWeights{}
		}
		moe.Router.GateInput = mt
	case "exp_probs_b.bias":
		if moe.Router == nil {
			moe.Router = &MOERouterWeights{}
		}
		moe.Router.ExpertProbBias = mt
	case "ffn_down_exps.weight":
		if moe.Experts == nil {
			moe.Experts = &MOEExpertWeights{}
		}
		moe.Experts.FfnDownExperts = mt
	case "ffn_up_exps.weight":
		if moe.Experts == nil {
			moe.Experts = &MOEExpertWeights{}
		}
		moe.Experts.FfnUpExperts = mt
	case "ffn_gate_exps.weight":
		if moe.Experts == nil {
			moe.Experts = &MOEExpertWeights{}
		}
		moe.Experts.FfnGateExperts = mt
	case "ffn_down_shexp.weight":
		if moe.Shared == nil {
			moe.Shared = &MOESharedWeights{}
		}
		moe.Shared.FfnDownShared = mt
	case "ffn_up_shexp.weight":
		if moe.Shared == nil {
			moe.Shared = &MOESharedWeights{}
		}
		moe.Shared.FfnUpShared = mt
	case "ffn_gate_shexp.weight":
		if moe.Shared == nil {
			moe.Shared = &MOESharedWeights{}
		}
		moe.Shared.FfnGateShared = mt
	}
}

func isNeededTensor(name string) bool {
	lowerName := strings.ToLower(name)

	if isGlobalWeight(lowerName, name) {
		return true
	}

	if strings.Contains(lowerName, "blk.") {
		return isLayerWeight(lowerName)
	}

	return false
}

func isGlobalWeight(lowerName, name string) bool {
	globalSuffixes := []string{
		"token_embd.weight",
		"output_norm.weight",
		"output.weight",
		"model.embed_tokens.weight",
		"model.norm.weight",
		"model.lm_head.weight",
	}

	for _, s := range globalSuffixes {
		if strings.HasSuffix(lowerName, s) && !strings.Contains(lowerName, "blk.") {
			return true
		}
	}
	return false
}

func isLayerWeight(lowerName string) bool {
	layerSuffixes := []string{
		"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_norm.weight",
		"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "ffn_norm.weight",
		"ssm_a", "ssm_d", "ssm_conv1d.weight", "ssm_conv1d.bias",
		"ssm_dt.weight", "ssm_dt.bias", "ssm_norm.weight", "ssm_norm.bias",
		"ssm_out.weight", "ssm_in.weight",
		"ffn_gate_inp.weight", "exp_probs_b.bias",
		"ffn_down_exps.weight", "ffn_up_exps.weight", "ffn_gate_exps.weight",
		"ffn_down_shexp.weight", "ffn_up_shexp.weight", "ffn_gate_shexp.weight",
	}

	for _, s := range layerSuffixes {
		if strings.HasSuffix(lowerName, s) {
			return true
		}
	}
	return false
}

func isNormWeight(name string) bool {
	normSuffixes := []string{
		"attn_norm.weight",
		"ffn_norm.weight",
		"output_norm.weight",
		"model.norm.weight",
		"ssm_norm.weight",
		"ssm_norm.bias",
	}
	lowerName := strings.ToLower(name)
	for _, s := range normSuffixes {
		if strings.HasSuffix(lowerName, s) {
			return true
		}
	}
	return false
}
