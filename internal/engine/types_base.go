package engine

type SamplerConfig struct {
	Temperature      float64
	TopK             int
	TopP             float64
	RepPenalty       float64 // 1.0 = no penalty, > 1.0 = penalty
	Seed             int64
	DebugActivations bool
	QualityMode      bool // Enable advanced sampling with nucleus sampling and adaptive temperature
}
