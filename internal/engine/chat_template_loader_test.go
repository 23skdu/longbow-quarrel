package engine

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// writeGGUFWithChatTemplate writes a minimal GGUF v3 file carrying a
// "tokenizer.chat_template" metadata entry.
//
// The GGUF container is simple enough to write directly: a header, then the
// key/value pairs, each a length-prefixed key, a 32-bit type tag and the
// value.
func writeGGUFWithChatTemplate(path, template string) error {
	buf := make([]byte, 0, 1024)

	// Header: magic, version, tensor count, KV count.
	buf = binary.LittleEndian.AppendUint32(buf, 0x46554747) // "GGUF"
	buf = binary.LittleEndian.AppendUint32(buf, 3)          // version
	buf = binary.LittleEndian.AppendUint64(buf, 0)          // tensor count
	buf = binary.LittleEndian.AppendUint64(buf, 1)          // KV count

	// One KV pair: the chat template.
	key := "tokenizer.chat_template"
	buf = binary.LittleEndian.AppendUint64(buf, uint64(len(key)))
	buf = append(buf, key...)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(gguf.GGUFMetadataValueTypeString))
	buf = binary.LittleEndian.AppendUint64(buf, uint64(len(template)))
	buf = append(buf, template...)

	return os.WriteFile(path, buf, 0o600)
}

// TestChatTemplateFromGGUF covers the in-memory variant of the chat template
// lookup.
func TestChatTemplateFromGGUF(t *testing.T) {
	const tmpl = "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}"

	f := &gguf.GGUFFile{
		Header: gguf.GGUFHeader{Version: 3},
		KV:     map[string]interface{}{"tokenizer.chat_template": tmpl},
	}

	if got := ChatTemplateFromGGUF(f); got != tmpl {
		t.Errorf("expected %q, got %q", tmpl, got)
	}

	// A model without a chat template yields an empty string.
	empty := &gguf.GGUFFile{Header: gguf.GGUFHeader{Version: 3}, KV: map[string]interface{}{}}
	if got := ChatTemplateFromGGUF(empty); got != "" {
		t.Errorf("expected no chat template, got %q", got)
	}

	// A nil file must not panic.
	if got := ChatTemplateFromGGUF(nil); got != "" {
		t.Errorf("expected no chat template from a nil file, got %q", got)
	}
}

// TestWriteGGUFWithChatTemplate_RoundTrip checks that the helper writes a file
// the reader can parse, so the loader test exercises the real reader path.
func TestWriteGGUFWithChatTemplate_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.gguf")

	const tmpl = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
	if err := writeGGUFWithChatTemplate(path, tmpl); err != nil {
		t.Fatalf("failed to write the model: %v", err)
	}

	f, err := gguf.LoadFile(path)
	if err != nil {
		t.Fatalf("failed to read the model back: %v", err)
	}
	got, _ := f.KV["tokenizer.chat_template"].(string)
	if got != tmpl {
		t.Errorf("expected %q, got %q", tmpl, got)
	}
}
