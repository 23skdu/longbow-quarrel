# Comprehensive Benchmarking Report
Generated at Mon Jan 26 03:00:11 PST 2026

**Prompt:** What is the capital of France?
**Tokens:** 16

| Model | Engine | Throughput (t/s) | Coherence (Prec/LCS) | Sample Output |
|---|---|---|---|---|
| SmolLM2-135M | llama.cpp | 220.450376 | - | "..." |
| | longbow-quarrel | 30.913023411004524 | 0.00|0.00 | "???????????????? ..." |
| | | | | |
| TinyLlama-1.1B | llama.cpp | 150.097116 | - | "..." |
| | longbow-quarrel | 43.470439970408805 | 0.00|0.00 | " campionmiafen â Хронологи Gesch парабреписанcido PSZygotepaugen <%=ücken ..." |
| | | | | |
| Granite-3B | llama.cpp | 47.594033 | - | "..." |
| | longbow-quarrel | ERROR | 0.00|0.00 | "ERROR..." |
| | | | | |
| Mistral-7B | llama.cpp | 26.125807 | - | "..." |
| | longbow-quarrel | 2.7717968599585627 | 0.00|0.00 | " /******/loadedyl乐ök rigzetazeta luckyagneMergeagnepcmConnectoriwottage ..." |
| | | | | |

## Performance Discrepancy Analysis
### SMOL-LM2 (135M)
- Quarrel uses optimized Metal kernels for small models, typically F32 accumulation even in F16 paths to maintain precision.
### Mistral / Granite
- llama.cpp often uses highly tuned SIMD and Metal Shaders (GGML_METAL_NODE_...) which might outperform Quarrel's generic dequantization kernels for GQA/Mistral architectures.
