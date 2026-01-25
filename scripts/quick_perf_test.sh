#!/bin/bash

echo "Quick performance test"
./bin/metal_benchmark -model "$HOME/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f" -prompt "test" -tokens 10 -temperature 0.7 -output json 2>/dev/null