# Comprehensive Benchmarking Report
Generated at Mon Jan 26 03:03:40 PST 2026

**Prompt:** What is the capital of France?
**Tokens:** 16

| Model | Engine | Throughput (t/s) | Coherence (Prec/LCS) | Sample Output |
|---|---|---|---|---|
| SmolLM2-135M | llama.cpp | 227.681719 | - | "..." |
| | longbow-quarrel | 28.465677714582252 | 0.00|0.00 | "???????????????? ..." |
| | | | | |
| TinyLlama-1.1B | llama.cpp | 143.302444 | - | "..." |
| | longbow-quarrel | 14.186753233194535 | 0.00|0.00 | "minipagemaste Становништвоubrella Википедииshift\<^izin∷lachtTDIB<0xAD> shall⁄ ..." |
| | | | | |
| Granite-3B | llama.cpp | 47.323405 | - | "..." |
| | longbow-quarrel | 8.815657909197057 | 0.00|0.00 | "poly831A1G7/='.C!7$ ..." |
| | | | | |
| Mistral-7B | llama.cpp | 26.083373 | - | "..." |
| | longbow-quarrel | 1.7397385948249977 | 0.00|0.00 | " /******/LECTzigxA indoorahanisten specobabiaERRemet franc teaamanscribed ..." |
| | | | | |

## Performance Discrepancy Analysis
### SMOL-LM2 (135M)
- Quarrel uses optimized Metal kernels for small models, typically F32 accumulation even in F16 paths to maintain precision.
### Mistral / Granite
- llama.cpp often uses highly tuned SIMD and Metal Shaders (GGML_METAL_NODE_...) which might outperform Quarrel's generic dequantization kernels for GQA/Mistral architectures.
