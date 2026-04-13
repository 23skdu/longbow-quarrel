package engine

import "github.com/23skdu/longbow-quarrel/internal/sampler"

type SamplerConfig struct {
	Temperature      float64
	TopK             int
	TopP             float64
	RepPenalty       float64
	Seed             int64
	DebugActivations bool
	QualityMode      bool
	SequenceID       uint64
	Grammar          *sampler.Grammar
}
