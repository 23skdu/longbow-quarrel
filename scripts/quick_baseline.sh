#!/bin/bash

echo "=== TinyLlama Baseline ==="
./bin/metal_benchmark -model /Users/rsd/.ollama/models/blobs/sha256-2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816 -tokens 100 -output json | tail -1

echo "=== Mistral Baseline ==="  
./bin/metal_benchmark -model /Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f -tokens 100 -output json | tail -1

echo "=== Granite4 Baseline ==="
./bin/metal_benchmark -model /Users/rsd/.ollama/models/blobs/sha256-5c7ac4aead1bcf44726e1d9601a9142208a42671984d3eba44dada7f8149501a0a0 -tokens 100 -output json | tail -1

echo "=== GPT-OSS Baseline ==="
./bin/metal_benchmark -model /Users/rsd/.ollama/models/blobs/sha256-e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb -tokens 100 -output json | tail -1