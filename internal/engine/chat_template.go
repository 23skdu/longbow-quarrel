package engine

import (
	"fmt"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// LoadGGUFChatTemplate reads the model's Jinja2 chat template from a GGUF
// file.
//
// The template lives in the metadata key "tokenizer.chat_template". It
// returns an empty string when the model does not ship one, and an error
// only when the file cannot be read.
func LoadGGUFChatTemplate(modelPath string) (string, error) {
	f, err := gguf.LoadFile(modelPath)
	if err != nil {
		return "", fmt.Errorf("failed to load GGUF %s: %w", modelPath, err)
	}
	defer func() { _ = f.Close() }()

	tmpl, _ := f.KV["tokenizer.chat_template"].(string)
	return tmpl, nil
}

// ChatTemplateFromGGUF is the same as LoadGGUFChatTemplate but works on an
// already-open GGUF file, so a caller that has the model loaded does not
// have to read it twice.
func ChatTemplateFromGGUF(f *gguf.GGUFFile) string {
	if f == nil {
		return ""
	}
	tmpl, _ := f.KV["tokenizer.chat_template"].(string)
	return tmpl
}
