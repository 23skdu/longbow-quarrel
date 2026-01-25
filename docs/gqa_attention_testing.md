# GQA Attention Testing and Metrics

This document outlines the testing strategy and Prometheus metrics implemented for the Grouped Query Attention (GQA) mechanism in Longbow-Quarrel.

## Unit Tests

Comprehensive unit tests have been developed in `internal/device/attention_gqa_test.go` to validate the GQA implementation. These tests cover:

-   **`TestAttention_GQA`**: Verifies correct KV head mapping for various GQA configurations (e.g., 4:2 grouping) and Multi-Query Attention scenarios. It compares GPU output against a reference CPU implementation.
-   **`TestAttention_GQA_Comprehensive`**: Runs GQA with a range of parameters (heads, KV heads, head dimensions, sequence lengths, positions) and compares GPU results against a CPU reference to ensure correctness across different configurations.
-   **`FuzzAttentionGQA`**: A fuzz test designed to explore the GQA implementation with randomized inputs, aiming to uncover edge cases and potential panics or incorrect behavior. It constrains inputs to avoid excessive resource usage and invalid configurations.

## Integration Tests

An integration test placeholder has been created in `internal/device/attention_gqa_integration_test.go`. This test is currently skipped but is intended to:

-   Load a small GQA-compatible model.
-   Run inference using the GQA mechanism.
-   Verify the output against expected results to ensure end-to-end correctness.

## Prometheus Metrics

New Prometheus metrics have been added to `internal/metrics/metrics.go` to monitor GQA performance and configuration:

-   **`GQAAttentionHeads`**: A histogram recording the number of attention heads used in GQA configurations.
-   **`GQAkvHeadsRatio`**: A histogram tracking the ratio of attention heads to KV heads, indicating the GQA grouping strategy.
-   **`GQAHeadDim`**: A histogram recording the dimension of each attention head.

These metrics will provide valuable insights into how GQA is being utilized and help identify potential performance bottlenecks or suboptimal configurations.

## Next Steps

-   Implement the actual integration test for GQA.
-   Update `docs/nextsteps.md` to reflect the completion of GQA testing tasks.
