package engine

import (
	"testing"
)

func FuzzTokenizerV5(f *testing.F) {
	f.Skip("Requires tokenizer files - set HF_TOKEN to run")

}

func FuzzGGUFLoading(f *testing.F) {
	f.Skip("Requires GGUF files - set GGUF_PATH to run")

}