# Implementation Plan: KV Cache Sharing & Concurrent Requests

## Problem Statement

The current longbow-quarrel architecture processes inference requests sequentially, preventing concurrent usage of the same model instance. This limits throughput and prevents true multi-user support.

## Current Architecture Limitations

1. **Single Request Processor**: `adapter_metal.go` uses a single goroutine (`processRequests`) that processes all requests sequentially
2. **Shared Cache State**: The `Engine` struct has a single `CachePos` integer that gets reset to 0 for each inference
3. **No Thread Safety**: The `Engine.Infer()` method is not designed for concurrent access

## Implementation Strategy

### Phase 1: Concurrent Request Processing

**Goal**: Allow multiple requests to be processed concurrently on the same model instance

**Changes Required**:

1. **Modify `adapter_metal.go`**:
   - Remove the single `processRequests` goroutine
   - Launch inference in a separate goroutine per request
   - Add proper synchronization for shared state

2. **Modify `Engine` struct**:
   - Add request-specific cache state tracking
   - Ensure thread-safe access to KV cache

**Files to Modify**:
- `cmd/webui/engine/adapter_metal.go`
- `cmd/webui/engine/adapter_cuda.go`
- `cmd/webui/engine/adapter_cpu.go`
- `internal/engine/types.go`
- `internal/engine/engine.go`

### Phase 2: Request-Specific KV Cache

**Goal**: Each request maintains its own KV cache position and state

**Approach A: Cache Pooling**
- Pre-allocate multiple KV cache instances
- Assign cache instance to each request
- Return cache to pool when request completes

**Approach B: Dynamic Cache Slices**
- Modify `TensorKVCache` to support multiple cache positions
- Track cache usage per request
- Manage cache cleanup

**Recommended**: Approach A (Cache Pooling) for simplicity and predictable memory usage

### Phase 3: Thread Safety

**Goal**: Ensure safe concurrent access to shared resources

**Changes**:
1. Add mutex protection for `engines` map access
2. Add per-engine mutex for inference operations
3. Ensure tokenizer access is thread-safe

## Implementation Order

### Step 1: Analyze and Design (COMPLETED)
- ✅ Analyzed current architecture
- ✅ Identified limitations
- ✅ Documented requirements

### Step 2: Modify Adapter for Concurrency
- [ ] Remove sequential `processRequests` loop
- [ ] Launch inference in goroutine per request
- [ ] Add request tracking

### Step 3: Implement Request-Specific Cache State
- [ ] Modify `Engine` struct to support multiple cache positions
- [ ] Add cache pooling mechanism
- [ ] Update `inferInternal` to use request-specific cache

### Step 4: Add Thread Safety
- [ ] Add mutex for engine access
- [ ] Ensure tokenizer thread-safety
- [ ] Test concurrent access

### Step 5: Testing and Validation
- [ ] Unit tests for concurrent access
- [ ] Load testing with multiple concurrent requests
- [ ] Performance benchmarking

## Success Criteria

1. **Functional**:
   - Multiple requests can be processed concurrently
   - Each request maintains independent cache state
   - No race conditions or cache corruption
   - Existing API compatibility maintained

2. **Performance**:
   - No significant performance degradation for single requests
   - Improved throughput with multiple concurrent requests
   - Memory usage remains reasonable

3. **Code Quality**:
   - Thread-safe implementation
   - Proper error handling
   - Clear documentation

## Risk Assessment

**High Risk**:
- Concurrent access to Metal GPU context
- Memory management with multiple cache instances
- Performance impact of locking

**Mitigation**:
- Thorough testing with race detector
- Performance profiling
- Gradual rollout with feature flags

## Timeline Estimate

- **Phase 1 (Concurrent Processing)**: 2-3 days
- **Phase 2 (Cache Pooling)**: 3-4 days
- **Phase 3 (Thread Safety)**: 1-2 days
- **Testing & Validation**: 2-3 days

**Total**: 8-12 days

## Dependencies

- Access to Metal GPU documentation for concurrent context usage
- Understanding of Metal autorelease pool behavior with multiple threads
- Performance testing infrastructure

## Next Immediate Actions

1. Create a prototype of cache pooling mechanism
2. Test concurrent Metal GPU context usage
3. Implement minimal concurrent adapter (proof of concept)
