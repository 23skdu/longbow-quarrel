#!/usr/bin/env python3
"""
Simple script to extract activation info from llama-cli verbose output.
Run llama-cli with --verbose flag and pipe output here.
"""

import sys
import json
import re

def parse_llama_output(lines):
    """Parse llama-cli verbose output to extract activation info"""
    
    activations = {
        "prompt": "",
        "tokens": [],
        "embedding": [],
        "layers": [],
        "final_logits": {}
    }
    
    for line in lines:
        # Try to extract useful info from verbose output
        # This is a placeholder - actual llama.cpp output format needs inspection
        
        # Look for token info
        if "token" in line.lower():
            print(f"Token line: {line.strip()}", file=sys.stderr)
        
        # Look for layer info  
        if "layer" in line.lower():
            print(f"Layer line: {line.strip()}", file=sys.stderr)
    
    return activations

if __name__ == "__main__":
    lines = sys.stdin.readlines()
    result = parse_llama_output(lines)
    print(json.dumps(result, indent=2))
