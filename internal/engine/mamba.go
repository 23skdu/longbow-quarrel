//go:build darwin && metal



package engine

import (
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/logger"
)



// detectMambaLayers determines the Mamba layer pattern from model metadata
func (e *metalEngine) detectMambaLayers(f *gguf.GGUFFile, log logger.Logger) {
	arch := e.config.Architecture

	// Check for explicit Mamba layer pattern in metadata
	if val, ok := f.KV[arch+".mamba_layers"]; ok {
		// Support both string patterns ("even", "odd", "all", "none") and layer counts
		switch v := val.(type) {
		case string:
			e.config.MambaLayerPattern = v
			log.Debug("Mamba layer pattern from metadata", "pattern", v)
			return
		case uint32:
			// Layer count: > 0 means hybrid, specific pattern depends on interleaving
			if v == uint32(e.config.Layers) {
				e.config.IsHybrid = true
				e.config.MambaLayerPattern = "all"
			} else if v > 0 {
				e.config.IsHybrid = true
				// Assume even pattern for hybrid models by default
				e.config.MambaLayerPattern = "even"
			}
			log.Debug("Mamba layer count from metadata", "count", v)
			return
		}
	}

	// Check for explicit hybrid architecture flag
	if val, ok := f.KV[arch+".is_hybrid"]; ok {
		if b, ok := val.(bool); ok && b {
			e.config.IsHybrid = true
			e.config.MambaLayerPattern = "even" // Default pattern for hybrid models
			log.Debug("Hybrid architecture detected from metadata")
			return
		}
	}

	// Detect based on architecture patterns
	switch arch {
	case "mamba":
		// Pure Mamba model
		e.config.MambaLayerPattern = "all"
		log.Debug("Pure Mamba model detected")
	case "nemotron", "nemo":
		// Nemotron models typically have hybrid architecture (Mamba + Transformer)
		// Pattern is typically every other layer starting from 0
		e.config.IsHybrid = true
		e.config.MambaLayerPattern = "even"
		log.Debug("Nemotron hybrid model detected", "pattern", "even")
	default:
		// Check if any layers have Mamba weights
		hasMamba := false
		for _, mamba := range e.weights.Mamba {
			if mamba != nil {
				hasMamba = true
				break
			}
		}
		if hasMamba {
			e.config.IsHybrid = true
			// Pattern will be determined by IsMambaLayer based on weights
			log.Debug("Hybrid model detected from Mamba weights")
		}
	}
}

// CountMambaLayers returns the number of Mamba layers in the model
func (e *metalEngine) CountMambaLayers() int {
	count := 0
	for i := 0; i < e.config.Layers; i++ {
		if e.IsMambaLayer(i) {
			count++
		}
	}
	return count
}

// IsMambaLayer checks if a layer index corresponds to a Mamba layer
// for a hybrid model. Uses config-based detection when available,
// falling back to weight-based detection for robustness.
func (e *metalEngine) IsMambaLayer(layerIdx int) bool {
	if layerIdx < 0 || layerIdx >= e.config.Layers {
		return false
	}

	// Config-based detection: check if architecture has Mamba layers
	if e.config.MambaLayerPattern != "" {
		switch e.config.MambaLayerPattern {
		case "all":
			// All layers are Mamba (pure Mamba model)
			return true
		case "even":
			// Every other layer starting from 0 (e.g., 0, 2, 4, ...)
			return layerIdx%2 == 0
		case "odd":
			// Every other layer starting from 1 (e.g., 1, 3, 5, ...)
			return layerIdx%2 == 1
		case "none":
			// No Mamba layers (pure Transformer)
			return false
		default:
			// Unknown pattern, fall back to weight-based
		}
	}

	// Hybrid model detection: check if this is a hybrid architecture
	if e.config.IsHybrid && layerIdx < len(e.weights.Mamba) {
		return e.weights.Mamba[layerIdx] != nil
	}

	// Fallback: check if Mamba weights exist for this layer
	if layerIdx < len(e.weights.Mamba) {
		return e.weights.Mamba[layerIdx] != nil
	}

	return false
}

