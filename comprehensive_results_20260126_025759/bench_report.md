# Comprehensive Benchmarking Report
Generated at Mon Jan 26 02:57:59 PST 2026

**Prompt:** What is the capital of France?
**Tokens:** 16

| Model | Engine | Throughput (t/s) | Coherence (Prec/LCS) | Sample Output |
|---|---|---|---|---|
| SmolLM2-135M | llama.cpp | 0.00 | - | "..." |
| | longbow-quarrel | 35.7841233033039 | 0.00|0.00 | "???????????????? ..." |
| | | | | |
| TinyLlama-1.1B | llama.cpp | 0.00 | - | "..." |
| | longbow-quarrel | ERROR | 0.00|0.00 | "ERROR..." |
| | | | | |
| Granite-3B | llama.cpp | 0.00 | - | "..." |
| | longbow-quarrel | ERROR | 0.00|0.00 | "ERROR..." |
| | | | | |
| Mistral-7B | llama.cpp | 0.00 | - | "..." |
| | longbow-quarrel | 1.6135938898616227 | 0.00|0.00 | "[INST] asserttotibr[control_443][control_25][/INST]нияer these let[control_322][/AVAILABLE_TOOLS] Au..." |
| | | | | |

## Performance Discrepancy Analysis
### SMOL-LM2 (135M)
- Quarrel uses optimized Metal kernels for small models, typically F32 accumulation even in F16 paths to maintain precision.
### Mistral / Granite
- llama.cpp often uses highly tuned SIMD and Metal Shaders (GGML_METAL_NODE_...) which might outperform Quarrel's generic dequantization kernels for GQA/Mistral architectures.
