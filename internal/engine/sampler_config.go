package engine

type SamplerConfig struct {
	Temperature      float64
	TopK             int
	TopP             float64
	RepPenalty       float64
	Seed             int64
	DebugActivations bool
	QualityMode      bool
	SequenceID       uint64
}
