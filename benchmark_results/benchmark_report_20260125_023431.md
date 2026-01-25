# LLM Performance Benchmark Report

**Generated:** Sun Jan 25 02:36:13 PST 2026  
**Model:** Mistral 7B (4.3GB GGUF)  
**Test Prompt:** "The quick brown fox jumps over the lazy dog"  
**Tokens Generated:** 100  
**Iterations:** 3  

## System Information

- **OS:** Darwin 25.2.0
- **Architecture:** arm64
- **Memory:** 36.0 GB
- **Go Version:** go version go1.25.6 darwin/arm64

## Benchmark Results

### longbow-quarrel (Metal GPU)

| Metric | Tokens/sec |
|--------|-----------|
| Average | [0;34mRunning longbow-quarrel (Metal)...[0m
  Iteration 1/3:   3.20 tokens/sec (31.00s)
  Iteration 2/3:   3.20 tokens/sec (31.00s)
  Iteration 3/3:   3.20 tokens/sec (31.00s)
  Average: 3.22 tokens/sec
  Min:  tokens/sec
  Max: 3.2 tokens/sec

3.22 |
| Minimum | [0;34mRunning longbow-quarrel (Metal)...[0m
  Iteration 1/3:   3.20 tokens/sec (31.00s)
  Iteration 2/3:   3.20 tokens/sec (31.00s)
  Iteration 3/3:   3.20 tokens/sec (31.00s)
  Average: 3.22 tokens/sec
  Min:  tokens/sec
  Max: 3.2 tokens/sec |
| Maximum | [0;34mRunning longbow-quarrel (Metal)...[0m
  Iteration 1/3:   3.20 tokens/sec (31.00s)
  Iteration 2/3:   3.20 tokens/sec (31.00s)
  Iteration 3/3:   3.20 tokens/sec (31.00s)
  Average: 3.22 tokens/sec
  Min:  tokens/sec
  Max: 3.2 tokens/sec

3.2 |

### llama.cpp

| Metric | Tokens/sec |
|--------|-----------|
| Average | [0;34mRunning llama.cpp...[0m
  Iteration 1/3:   0.00 tokens/sec (0.00s)
  Iteration 2/3:   0.00 tokens/sec (0.00s)
  Iteration 3/3:   0.00 tokens/sec (0.00s)
  Average:  tokens/sec
  Min:  tokens/sec
  Max:  tokens/sec |
| Minimum | [0;34mRunning llama.cpp...[0m
  Iteration 1/3:   0.00 tokens/sec (0.00s)
  Iteration 2/3:   0.00 tokens/sec (0.00s)
  Iteration 3/3:   0.00 tokens/sec (0.00s)
  Average:  tokens/sec
  Min:  tokens/sec
  Max:  tokens/sec |
| Maximum | [0;34mRunning llama.cpp...[0m
  Iteration 1/3:   0.00 tokens/sec (0.00s)
  Iteration 2/3:   0.00 tokens/sec (0.00s)
  Iteration 3/3:   0.00 tokens/sec (0.00s)
  Average:  tokens/sec
  Min:  tokens/sec
  Max:  tokens/sec |

## Performance Comparison

| Implementation | Average Tokens/sec | Relative Performance |
|----------------|------------------|---------------------|
| longbow-quarrel (Metal GPU) | [0;34mRunning longbow-quarrel (Metal)...[0m
  Iteration 1/3:   3.20 tokens/sec (31.00s)
  Iteration 2/3:   3.20 tokens/sec (31.00s)
  Iteration 3/3:   3.20 tokens/sec (31.00s)
  Average: 3.22 tokens/sec
  Min:  tokens/sec
  Max: 3.2 tokens/sec

3.22 | N/Ax |
| llama.cpp | [0;34mRunning llama.cpp...[0m
  Iteration 1/3:   0.00 tokens/sec (0.00s)
  Iteration 2/3:   0.00 tokens/sec (0.00s)
  Iteration 3/3:   0.00 tokens/sec (0.00s)
  Average:  tokens/sec
  Min:  tokens/sec
  Max:  tokens/sec | 1.0x (baseline) |

