package engine

import (
	"os"
	"testing"
	"github.com/23skdu/longbow-quarrel/internal/config"
)

func TestActivationLogger_Coverage(t *testing.T) {
	logger := NewActivationLogger()
	logger.Enable("test prompt", []int{1, 2})
	if !logger.IsEnabled() {
		t.Error("Logger should be enabled")
	}

	data := []float32{1.0, 2.0}
	logger.LogEmbedding(data)
	logger.LogLayer(0, 1.0, 1.0, 1.0, 1.0, 1.0, data, data, data, data, data, data, data, data)
	logger.LogLogits(data, []int{0})

	tmp, _ := os.CreateTemp("", "act_log.txt")
	defer os.Remove(tmp.Name())
	logger.SaveToFile(tmp.Name())
}

func TestMockEngine_Coverage(t *testing.T) {
	cfg := config.Config{VocabSize: 100}
	e, err := NewMockEngine("mock_path", cfg)
	if err != nil { t.Errorf("NewMockEngine failed: %v", err) }
	
	e.Config()
	e.Close()
	e.SwapModel("new_path", cfg)
	e.GetSeqCachePos("seq_1")
	e.ForwardDraft([]int{1, 2})
	e.RollbackKV("seq_1", 0)
}
