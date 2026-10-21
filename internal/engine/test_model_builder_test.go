package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/testmodel"
)

// buildTestGGUF writes a small but complete llama-style model.
//
// The builder lives in the testmodel package so that the CUDA integration
// tests, which are a separate package, can use it too.
func buildTestGGUF(t testing.TB, dir string) string {
	t.Helper()
	return testmodel.Build(t, dir)
}
