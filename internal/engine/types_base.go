//go:build (linux && !cuda) || (linux && cuda)
// +build linux,!cuda linux,cuda

package engine

import (
	"github.com/23skdu/longbow-quarrel/internal/config"
)

func NewEngine(modelPath string, cfg config.Config) (Engine, error) {
	for _, creator := range engineCreators {
		engine, err := creator(modelPath, cfg)
		if err == nil {
			return engine, nil
		}
	}
	return nil, nil
}
