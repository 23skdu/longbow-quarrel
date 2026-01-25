# Docker Usage

This repository includes Docker support for Linux environments.

## Important Notes

- **Metal acceleration is macOS-only**: The full LLM inference requires Apple Silicon with Metal framework
- **Linux Docker builds**: Provide CPU-only functionality for GGUF file inspection and tokenizer testing
- **Cross-platform**: Works on any system with Docker installed

## Building the Docker Image

```bash
docker build -t longbow-quarrel .
```

## Running the Docker Container

The Docker image provides a CPU-only demo that can:
- Read GGUF file metadata and tensor information
- Demonstrate tokenizer functionality
- Work with any GGUF model files

```bash
# Show usage
docker run --rm longbow-quarrel

# Analyze a GGUF file (mount your model file)
docker run --rm -v /path/to/your/model.gguf:/model.gguf longbow-quarrel /model.gguf
```

## Example Output

```
=== Longbow Quarrel Linux Demo ===
Reading GGUF file: /model.gguf
Found 285 tensors
Metadata keys: 42
Architecture: llama
Vocab size: 32000
Context length: 8192
File type: 1

=== Tokenizer Demo ===
Tokenizer model: tokenizer.ggml.model
Test text: 'Hello, world!'
Tokens: [1, 15043, 3186, 2991]
Token count: 4

=== Note ===
This is a CPU-only demo for Linux environments.
For full Metal acceleration, use macOS with Apple Silicon.
```

## Full Feature Usage

For complete LLM inference with Metal GPU acceleration, use:
- macOS with Apple Silicon (M1/M2/M3/M4)
- Native Go build: `CGO_ENABLED=1 go run ./cmd/quarrel`
- Ollama model resolution and GGUF loading with GPU acceleration

## Development

To build the demo locally for testing:
```bash
CGO_ENABLED=0 GOOS=linux go build -o demo ./cmd/demo
```