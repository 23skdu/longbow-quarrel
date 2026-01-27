// go:build darwin && metal

#import "metal_bridge.h"
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface MetalWrapper : NSObject
@property(strong) id<MTLDevice> device;
@property(strong) id<MTLCommandQueue> commandQueue;
@property(strong) id<MTLLibrary> library;
@property(strong) id<MTLComputePipelineState> pipelineRMSNorm_F16;
@property(strong) id<MTLComputePipelineState> pipelineMatMul_F16;
@property(strong) id<MTLComputePipelineState> pipelineMatMul_Q4K_F16;
@property(strong) id<MTLComputePipelineState> pipelineMatMul_Q3K_F16;
@property(strong) id<MTLComputePipelineState> pipelineMatMul_Q6K_F16;
@property(strong) id<MTLComputePipelineState> pipelineRoPE_F16;
@property(strong) id<MTLComputePipelineState> pipelineEmbedding_F16;
@property(strong) id<MTLComputePipelineState> pipelineEmbedding_Q4K;
@property(strong) id<MTLComputePipelineState> pipelineEmbedding_Q4K_Optimized;
@property(strong) id<MTLComputePipelineState> pipelineStoreKV_F16;
@property(strong) id<MTLComputePipelineState> pipelineStoreKV_F16_Batch;
@property(strong) id<MTLComputePipelineState> pipelineAdd_F16;
@property(strong) id<MTLComputePipelineState> pipelineCopy_F16;
@property(strong) id<MTLComputePipelineState> pipelineSwiGLU_F16;
@property(strong) id<MTLComputePipelineState> pipelineSoftmax_F16;
@property(strong) id<MTLComputePipelineState> pipelineAttScores_F16;
@property(strong) id<MTLComputePipelineState> pipelineAttValues_F16;
@property(strong) id<MTLComputePipelineState> pipelineAttFused_F16;
@property(strong) id<MTLComputePipelineState> pipelineAttPaged_F16;
@property(strong) id<MTLComputePipelineState> pipelineScale_F16;
// FP32 Pipelines
@property(strong) id<MTLComputePipelineState> pipelineRMSNorm_F32;
@property(strong) id<MTLComputePipelineState> pipelineMatMul_Q4K_F32;
@property(strong) id<MTLComputePipelineState> pipelineMatMul_Q6K_F32;
@property(strong) id<MTLComputePipelineState> pipelineLinearQ6K_F16_F32;
@property(strong) id<MTLComputePipelineState> pipelineMatMul_Q4K_F32_F16;
@property(strong) id<MTLComputePipelineState> pipelineMatMul_Q6K_F32_F16;
@property(strong) id<MTLComputePipelineState> pipelineAdd_F32;
@property(strong) id<MTLComputePipelineState> pipelineSwiGLU_F32;
// @property(strong) id<MTLComputePipelineState> pipelineSwiGLULinear_Q4K_F16;
// // Already declared
@property(strong) id<MTLComputePipelineState> pipelineFillZero;
@property(strong) id<MTLComputePipelineState> pipelineMatMul_F16_F32;
@property(strong) id<MTLComputePipelineState> pipelineCopy_F16_F32;
// FP32 FFN Pipelines for Small Models
@property(strong) id<MTLComputePipelineState> pipelineLinearF16ToF32;
@property(strong) id<MTLComputePipelineState> pipelineLinearF32ToF16;
@property(strong) id<MTLComputePipelineState> pipelineCopy_F32_F16;
// Specialized Mixed Precision
@property(strong) id<MTLComputePipelineState> pipelineMatMul_F16_F16_F32;
// Mixed Precision
@property(strong) id<MTLComputePipelineState> pipelineRMSNorm_F32_F16;
@property(strong) id<MTLComputePipelineState> pipelineAdd_F32_F16;
@property(strong) id<MTLComputePipelineState> pipelineRMSNormLinear_Q4K_F16;
@property(strong) id<MTLComputePipelineState> pipelineRMSNormLinear_F16;
@property(strong) id<MTLComputePipelineState> pipelineRMSNormQKV_F16;
@property(strong) id<MTLComputePipelineState> pipelineRMSNormQKV_Q4K_F16;
@property(strong) id<MTLComputePipelineState> pipelineDebugRoPEFreq;
@property(strong) id<MTLComputePipelineState> pipelineDebugDot;
@property(strong) id<MTLComputePipelineState> pipelineSwiGLULinear_Q4K_F16;
// Q4_0
@property(strong) id<MTLComputePipelineState> pipelineLinearQ4_0_F16;
@property(strong) id<MTLComputePipelineState> pipelineLinearQ4_0_F32;
@property(strong) id<MTLComputePipelineState> pipelineEmbeddingQ4_0_F16;
// Mamba / SSM
@property(strong) id<MTLComputePipelineState> pipelineMambaConv1d_F16;
@property(strong) id<MTLComputePipelineState> pipelineMambaScan_F16;
@property(strong) id<MTLComputePipelineState> pipelineSiLU_F16;
@property(strong) id<MTLComputePipelineState> pipelineSlice_F16;
@property(strong) id<MTLComputePipelineState> pipelineMul_F16;
@property(strong) id<MTLCommandBuffer> currentCommandBuffer;
@property(strong) id<MTLComputeCommandEncoder> currentEncoder;
@property(strong) id<MTLCommandBuffer> lastCommandBuffer;
@end

@implementation MetalWrapper
- (id<MTLComputeCommandEncoder>)ensureEncoder {
  if (!self.currentEncoder) {
    if (!self.currentCommandBuffer)
      self.currentCommandBuffer = [self.commandQueue commandBuffer];
    self.currentEncoder = [self.currentCommandBuffer computeCommandEncoder];
  }
  return self.currentEncoder;
}
- (void)flush {
  if (self.currentEncoder) {
    [self.currentEncoder endEncoding];
    self.currentEncoder = nil;
  }
  if (self.currentCommandBuffer) {
#if !__has_feature(objc_arc)
    if (self.lastCommandBuffer)
      [self.lastCommandBuffer release];
    self.lastCommandBuffer = [self.currentCommandBuffer retain];
#else
    self.lastCommandBuffer = self.currentCommandBuffer;
#endif
    [self.currentCommandBuffer commit];
    self.currentCommandBuffer = nil;
  }
}
- (void)synchronize {
  [self flush];
  if (self.lastCommandBuffer) {
    [self.lastCommandBuffer waitUntilCompleted];
#if !__has_feature(objc_arc)
    [self.lastCommandBuffer release];
#endif
    self.lastCommandBuffer = nil;
  }
}
- (void)barrier {
  [[self ensureEncoder] memoryBarrierWithScope:MTLBarrierScopeBuffers];
}
@end

static id<MTLComputePipelineState> loadPipeline(MetalWrapper *ctx,
                                                NSString *name) {
  id<MTLFunction> f = [ctx.library newFunctionWithName:name];
  if (!f) {
    fprintf(stderr, "Metal Error: Function '%s' not found in library\n",
            [name UTF8String]);
    fflush(stderr);
    return nil;
  }
  NSError *err = nil;
  id<MTLComputePipelineState> p =
      [ctx.device newComputePipelineStateWithFunction:f error:&err];
  if (!p) {
    fprintf(stderr, "Metal Error: Failed to create pipeline '%s': %s\n",
            [name UTF8String], [[err localizedDescription] UTF8String]);
    fflush(stderr);
    return nil;
  }
  return p;
}

// Autorelease Pool
void *Metal_AutoreleasePoolPush() {
#if __has_feature(objc_arc)
  return (__bridge void *)[NSAutoreleasePool new];
#else
  return [NSAutoreleasePool new];
#endif
}

void Metal_AutoreleasePoolPop(void *pool) {
#if __has_feature(objc_arc)
  [(__bridge NSAutoreleasePool *)pool drain];
#else
  [(NSAutoreleasePool *)pool drain];
#endif
}

MetalContextRef Metal_Init(const char *libSource) {
  MetalWrapper *ctx = [[MetalWrapper alloc] init];
  ctx.device = MTLCreateSystemDefaultDevice();
  if (!ctx.device) {
    fprintf(stderr, "Metal_Init Error: Failed to create device\n");
    fflush(stderr);
    return NULL;
  }
  ctx.commandQueue = [ctx.device newCommandQueue];
  if (!ctx.commandQueue) {
    fprintf(stderr, "Metal_Init Error: Failed to create command queue\n");
    fflush(stderr);
    return NULL;
  }
  NSError *err = nil;
  ctx.library =
      [ctx.device newLibraryWithSource:[NSString stringWithUTF8String:libSource]
                               options:nil
                                 error:&err];
  if (!ctx.library) {
    fprintf(stderr, "Metal_Init Error (Library): %s\n",
            [[err localizedDescription] UTF8String]);
    fflush(stderr);
    return NULL;
  }
  ctx.pipelineRMSNorm_F16 = loadPipeline(ctx, @"rmsnorm_f16");
  ctx.pipelineMatMul_F16 = loadPipeline(ctx, @"linear_f16");
  ctx.pipelineMatMul_Q4K_F16 = loadPipeline(ctx, @"linear_q4k_f16");
  ctx.pipelineMatMul_Q3K_F16 = loadPipeline(ctx, @"linear_q3k_f16");
  ctx.pipelineMatMul_Q6K_F16 = loadPipeline(ctx, @"linear_q6k_f16");
  ctx.pipelineRoPE_F16 = loadPipeline(ctx, @"rope_f16");
  ctx.pipelineEmbedding_F16 = loadPipeline(ctx, @"embedding_f16");
  ctx.pipelineEmbedding_Q4K = loadPipeline(ctx, @"embedding_q4k_f16");
  ctx.pipelineEmbedding_Q4K_Optimized =
      loadPipeline(ctx, @"embedding_q4k_f16_optimized");
  ctx.pipelineStoreKV_F16 = loadPipeline(ctx, @"store_kv_f16");
  ctx.pipelineStoreKV_F16_Batch = loadPipeline(ctx, @"store_kv_f16_batch");
  ctx.pipelineAdd_F16 = loadPipeline(ctx, @"add_f16");
  ctx.pipelineCopy_F16 = loadPipeline(ctx, @"copy_f16");
  ctx.pipelineSwiGLU_F16 = loadPipeline(ctx, @"swiglu_f16");
  ctx.pipelineSoftmax_F16 = loadPipeline(ctx, @"softmax_f16");
  ctx.pipelineAttScores_F16 = loadPipeline(ctx, @"att_scores_f16_v2");
  ctx.pipelineAttValues_F16 = loadPipeline(ctx, @"att_values_f16");
  ctx.pipelineAttFused_F16 = loadPipeline(ctx, @"att_fused_f16");
  ctx.pipelineAttPaged_F16 = loadPipeline(ctx, @"att_paged_f16");
  ctx.pipelineScale_F16 = loadPipeline(ctx, @"scale_f16");
  ctx.pipelineRMSNormLinear_F16 = loadPipeline(ctx, @"rmsnorm_linear_f16");
  ctx.pipelineRMSNormQKV_F16 = loadPipeline(ctx, @"rmsnorm_qkv_f16");
  ctx.pipelineRMSNormQKV_Q4K_F16 = loadPipeline(ctx, @"rmsnorm_qkv_q4k_f16");

  // FP32 Load
  ctx.pipelineRMSNorm_F32 = loadPipeline(ctx, @"rmsnorm_f32");
  ctx.pipelineMatMul_Q4K_F32 = loadPipeline(ctx, @"linear_q4k_f32");
  ctx.pipelineMatMul_Q6K_F32 = loadPipeline(ctx, @"linear_q6k_f32");
  ctx.pipelineLinearQ6K_F16_F32 = loadPipeline(ctx, @"linear_q6k_f16_f32");
  ctx.pipelineMatMul_Q4K_F32_F16 = loadPipeline(ctx, @"linear_q4k_f32_f16");
  ctx.pipelineMatMul_Q6K_F32_F16 = loadPipeline(ctx, @"linear_q6k_f32_f16");
  ctx.pipelineAdd_F32 = loadPipeline(ctx, @"add_f32");
  ctx.pipelineSwiGLU_F32 = loadPipeline(ctx, @"swiglu_f32");
  ctx.pipelineSwiGLULinear_Q4K_F16 =
      loadPipeline(ctx, @"swiglu_linear_q4k_f16");
  // Q4_0
  ctx.pipelineLinearQ4_0_F16 = loadPipeline(ctx, @"linear_q4_0_f16");
  ctx.pipelineLinearQ4_0_F32 = loadPipeline(ctx, @"linear_q4_0_f32");
  ctx.pipelineEmbeddingQ4_0_F16 = loadPipeline(ctx, @"embedding_q4_0_f16");
  ctx.pipelineMatMul_F16_F32 = loadPipeline(ctx, @"linear_f16_f32");
  ctx.pipelineCopy_F16_F32 = loadPipeline(ctx, @"copy_f16_to_f32");
  ctx.pipelineCopy_F32_F16 = loadPipeline(ctx, @"copy_f32_to_f16");
  ctx.pipelineMatMul_F16_F16_F32 =
      loadPipeline(ctx, @"linear_f16_in_f16_out_f32");

  // FP32 FFN Pipelines for Small Models
  ctx.pipelineLinearF16ToF32 = loadPipeline(ctx, @"linear_f16_to_f32");
  ctx.pipelineLinearF32ToF16 = loadPipeline(ctx, @"linear_f32_to_f16");
  ctx.pipelineRMSNorm_F32_F16 = loadPipeline(ctx, @"rmsnorm_f32_to_f16_v4");
  ctx.pipelineAdd_F32_F16 = loadPipeline(ctx, @"add_f32_f16");

  ctx.pipelineFillZero = loadPipeline(ctx, @"fill_zero"); // New Kernel
  ctx.pipelineRMSNormLinear_Q4K_F16 =
      loadPipeline(ctx, @"rmsnorm_linear_q4k_f16");
  ctx.pipelineDebugRoPEFreq = loadPipeline(ctx, @"debug_rope_freq");
  ctx.pipelineDebugDot = loadPipeline(ctx, @"debug_dot");
  ctx.pipelineSwiGLULinear_Q4K_F16 =
      loadPipeline(ctx, @"swiglu_linear_q4k_f16");

  // Mamba / SSM
  ctx.pipelineMambaConv1d_F16 = loadPipeline(ctx, @"mamba_conv1d_f16");
  ctx.pipelineMambaScan_F16 = loadPipeline(ctx, @"mamba_scan_step_f16");
  ctx.pipelineSiLU_F16 = loadPipeline(ctx, @"silu_f16");
  ctx.pipelineSlice_F16 = loadPipeline(ctx, @"slice_f16");
  ctx.pipelineMul_F16 = loadPipeline(ctx, @"mul_f16");

#if __has_feature(objc_arc)
  return (__bridge_retained MetalContextRef)ctx;
#else
  return (MetalContextRef)ctx;
#endif
}

void Metal_Free(MetalContextRef ctx) {
#if __has_feature(objc_arc)
  MetalWrapper *mc = (__bridge_transfer MetalWrapper *)ctx;
#else
  MetalWrapper *mc = (MetalWrapper *)ctx;
#endif
  [mc flush];
}
void Metal_Synchronize(MetalContextRef ctx) {
#if __has_feature(objc_arc)
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
#else
  MetalWrapper *mc = (MetalWrapper *)ctx;
#endif
  [mc synchronize];
}
MetalBufferRef Metal_Alloc(MetalContextRef ctx, long long size) {
#if __has_feature(objc_arc)
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
#else
  MetalWrapper *mc = (MetalWrapper *)ctx;
#endif
  id<MTLBuffer> buf =
      [[mc device] newBufferWithLength:size
                               options:MTLResourceStorageModeShared];
#if !__has_feature(objc_arc)
  [buf retain];
#endif
#if __has_feature(objc_arc)
  return (__bridge_retained MetalBufferRef)buf;
#else
  return (MetalBufferRef)buf;
#endif
}

MetalBufferRef Metal_AllocPrivate(MetalContextRef ctx, long long size) {
#if __has_feature(objc_arc)
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
#else
  MetalWrapper *mc = (MetalWrapper *)ctx;
#endif
  id<MTLBuffer> buf =
      [[mc device] newBufferWithLength:size
                               options:MTLResourceStorageModePrivate];
#if !__has_feature(objc_arc)
  [buf retain];
#endif
#if __has_feature(objc_arc)
  return (__bridge_retained MetalBufferRef)buf;
#else
  return (MetalBufferRef)buf;
#endif
}
void Metal_FreeBuffer(MetalContextRef ctx, MetalBufferRef buf) {
#if __has_feature(objc_arc)
  id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf;
#else
  id<MTLBuffer> b = (id<MTLBuffer>)buf;
#endif
#if !__has_feature(objc_arc)
  [b release];
#endif
  b = nil;
}
// Heap Allocation
void *Metal_NewHeap(MetalContextRef ctx, long long size) {
  MTLHeapDescriptor *desc = [MTLHeapDescriptor new];
  desc.size = size;
  desc.storageMode = MTLStorageModeShared; // CPU accessible
  desc.cpuCacheMode = MTLCPUCacheModeDefaultCache;
  desc.hazardTrackingMode = MTLHazardTrackingModeDefault;

  id<MTLHeap> heap =
#if __has_feature(objc_arc)
      [[(__bridge MetalWrapper *)ctx device] newHeapWithDescriptor:desc];
#else
      [[(MetalWrapper *)ctx device] newHeapWithDescriptor:desc];
#endif
  if (!heap) {
    printf("ERROR: Failed to allocate Metal Heap of size %lld\n", size);
    return NULL;
  }
#if !__has_feature(objc_arc)
  [heap retain];
  return (void *)heap;
#else
  return (__bridge_retained void *)heap;
#endif
}

MetalBufferRef Metal_NewBufferFromHeap(void *heapRef, long long size) {
#if __has_feature(objc_arc)
  id<MTLHeap> heap = (__bridge id<MTLHeap>)heapRef;
#else
  id<MTLHeap> heap = (id<MTLHeap>)heapRef;
#endif
  id<MTLBuffer> buf = [heap newBufferWithLength:size
                                        options:MTLResourceStorageModeShared];
  if (!buf) {
    printf("ERROR: Failed to allocate Buffer from Heap size %lld\n", size);
    return NULL;
  }
#if !__has_feature(objc_arc)
  [buf retain];
#endif
#if __has_feature(objc_arc)
  return (__bridge_retained MetalBufferRef)buf;
#else
  return (MetalBufferRef)buf;
#endif
}

void Metal_FreeHeap(void *heap) {
#if __has_feature(objc_arc)
  id<MTLHeap> h = (__bridge_transfer id<MTLHeap>)heap;
#else
  id<MTLHeap> h = (id<MTLHeap>)heap;
#endif
#if !__has_feature(objc_arc)
  [h release];
#endif
  h = nil;
}

void Metal_CopyToDevice(MetalBufferRef buf, int o, const void *d, int s) {
  id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf;
  memcpy([b contents] + o, d, s);
#if defined(__MAC_10_15) || defined(__IPHONE_13_0)
  [b didModifyRange:NSMakeRange(o, s)];
#endif
}
void Metal_CopyToHost(MetalBufferRef buf, int o, void *d, int s) {
  memcpy(d, [(__bridge id<MTLBuffer>)buf contents] + o, s);
}
void *Metal_GetBufferContents(MetalBufferRef buf) {
  return [(__bridge id<MTLBuffer>)buf contents];
}
void Metal_ZeroBuffer(MetalBufferRef buf, int o, int s) {
  id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf;
  memset([b contents] + o, 0, s);
  [b didModifyRange:NSMakeRange(o, s)];
}

void Metal_ZeroBufferGPU(MetalContextRef ctx, MetalBufferRef buf, int o,
                         int s) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineFillZero];
  [enc setBuffer:(__bridge id<MTLBuffer>)buf offset:o atIndex:0];
  [enc setBytes:&s length:4 atIndex:1];

  MTLSize gridSize = MTLSizeMake(s, 1, 1);
  MTLSize groupSize = MTLSizeMake(1024, 1, 1);
  if (gridSize.width < 1024)
    groupSize.width = gridSize.width;

  [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
}

void Metal_Embedding_F16(MetalContextRef ctx, MetalBufferRef weights, int offW,
                         MetalBufferRef result, int offRes, int rowIdx,
                         int cols) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineEmbedding_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)weights offset:offW atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)result offset:offRes atIndex:1];
  [enc setBytes:&rowIdx length:4 atIndex:2];
  [enc setBytes:&cols length:4 atIndex:3];
  [enc dispatchThreads:MTLSizeMake(cols, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(cols, 256), 1, 1)];
  [mc barrier];
}

void Metal_DebugRoPEFreq(MetalContextRef ctx, MetalBufferRef output,
                         int headDim, float theta, int pos) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineDebugRoPEFreq];
  [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:0];
  [enc setBytes:&headDim length:4 atIndex:1];
  [enc setBytes:&theta length:4 atIndex:2];
  [enc setBytes:&pos length:4 atIndex:3];

  // Launch threads for HeadDim/2 (pairs)
  // Actually kernel uses i = tid, tid < headDim/2.
  // So we launch headDim/2 threads.
  int threads = headDim / 2;
  [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(threads, 32), 1, 1)];
  [mc barrier];
}

void Metal_DebugDot(MetalContextRef ctx, MetalBufferRef a, MetalBufferRef b,
                    MetalBufferRef output, int dim) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineDebugDot];
  [enc setBuffer:(__bridge id<MTLBuffer>)a offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)b offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:2];
  [enc setBytes:&dim length:4 atIndex:3];

  // Single group, 32 threads
  [enc dispatchThreads:MTLSizeMake(32, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [mc barrier];
}

void Metal_Embedding_Q4K(MetalContextRef ctx, MetalBufferRef weights, int offW,
                         MetalBufferRef result, int offRes, int rowIdx,
                         int cols, float scale) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineEmbedding_Q4K];
  [enc setBuffer:(__bridge id<MTLBuffer>)weights offset:offW atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)result offset:offRes atIndex:1];
  [enc setBytes:&rowIdx length:4 atIndex:2];
  [enc setBytes:&cols length:4 atIndex:3];
  [enc setBytes:&scale length:4 atIndex:4];
  [enc dispatchThreads:MTLSizeMake(1024, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  [mc barrier];
}

void Metal_Embedding_Q4K_Optimized(MetalContextRef ctx, MetalBufferRef weights,
                                   int offW, MetalBufferRef result, int offRes,
                                   int rowIdx, int cols, float scale) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineEmbedding_Q4K_Optimized];
  [enc setBuffer:(__bridge id<MTLBuffer>)weights offset:offW atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)result offset:offRes atIndex:1];
  [enc setBytes:&rowIdx length:4 atIndex:2];
  [enc setBytes:&cols length:4 atIndex:3];
  [enc setBytes:&scale length:4 atIndex:4];
  // Use 1 thread per output element for better parallelism
  [enc dispatchThreads:MTLSizeMake(cols, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [mc barrier];
}

void Metal_RMSNorm_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                       MetalBufferRef weight, int offWeight,
                       MetalBufferRef result, int offRes, int rows, int cols,
                       float eps) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineRMSNorm_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:offIn atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)result offset:offRes atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:offWeight atIndex:2];
  [enc setBytes:&eps length:4 atIndex:3];
  [enc setBytes:&cols length:4 atIndex:4];
  // One threadgroup per row to share memory. Max 1024 columns supported.
  int threads = (cols < 1024) ? cols : 1024;
  [enc dispatchThreads:MTLSizeMake(threads, rows, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [mc barrier];
}

void Metal_MatMul_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                      bool transA, MetalBufferRef b, int offB, bool transB,
                      MetalBufferRef c, int offC, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineMatMul_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)b offset:offB atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)c offset:offC atIndex:2];
  [enc setBytes:&K length:4 atIndex:3];
  [enc setBytes:&N length:4 atIndex:4];
  // 32 threads per row (1 SIMD group)
  [enc dispatchThreads:MTLSizeMake(32, N, M)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
}

void Metal_MatMul_Q4K_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                          bool transA, MetalBufferRef b, int offB, bool transB,
                          MetalBufferRef c, int offC, int M, int N, int K,
                          float scale) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineMatMul_Q4K_F16];

  [enc setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)b offset:offB atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)c offset:offC atIndex:2];
  [enc setBytes:&K length:4 atIndex:3];
  [enc setBytes:&N length:4 atIndex:4];
  [enc setBytes:&scale length:4 atIndex:5];

  [enc dispatchThreads:MTLSizeMake(32, N, M)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
}

void Metal_MatMul_Q3K_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                          bool transA, MetalBufferRef b, int offB, bool transB,
                          MetalBufferRef c, int offC, int M, int N, int K,
                          float scale) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineMatMul_Q3K_F16];

  [enc setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)b offset:offB atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)c offset:offC atIndex:2];
  [enc setBytes:&K length:4 atIndex:3];
  [enc setBytes:&N length:4 atIndex:4];
  [enc setBytes:&scale length:4 atIndex:5];

  [enc dispatchThreads:MTLSizeMake(32, N, M)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
}

void Metal_MatMul_Q6K_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                          bool transA, MetalBufferRef b, int offB, bool transB,
                          MetalBufferRef c, int offC, int M, int N, int K,
                          float scale) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineMatMul_Q6K_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)b offset:offB atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)c offset:offC atIndex:2];
  [enc setBytes:&K length:4 atIndex:3];
  [enc setBytes:&N length:4 atIndex:4];
  [enc setBytes:&scale length:4 atIndex:5];
  [enc dispatchThreads:MTLSizeMake(32, N, M)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
}

void Metal_Add_F16(MetalContextRef ctx, MetalBufferRef a, int oA,
                   MetalBufferRef b, int oB, MetalBufferRef r, int oR,
                   int count) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineAdd_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)a offset:oA atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)b offset:oB atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)r offset:oR atIndex:2];
  [enc dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(count, 256), 1, 1)];
  [mc barrier];
}

void Metal_Scale_F16(MetalContextRef ctx, MetalBufferRef x, int offX,
                     float scale, MetalBufferRef result, int offRes,
                     int count) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineScale_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)x offset:offX atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)result offset:offRes atIndex:1];
  [enc setBytes:&scale length:sizeof(float) atIndex:2];
  [enc dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(count, 256), 1, 1)];
  [mc barrier];
}
void Metal_RoPE_F16(MetalContextRef ctx, MetalBufferRef d, int oD, int b, int s,
                    int nh, int hd, int po, float rt) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineRoPE_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)d offset:oD atIndex:0];
  [enc setBytes:&po length:4 atIndex:1];
  [enc setBytes:&hd length:4 atIndex:2];
  [enc setBytes:&rt length:4 atIndex:3];
  [enc setBytes:&nh length:4 atIndex:4];

  int pairs = nh * (hd / 2);
  // Dispatch: (pairs, seqLen, 1)
  // kernel uses uint2 gid
  [enc dispatchThreads:MTLSizeMake(pairs, s, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(pairs, 256), 1, 1)];
  [mc barrier];
}

void Metal_SwiGLU_F16(MetalContextRef ctx, MetalBufferRef iV, int oV,
                      MetalBufferRef iG, int oG, MetalBufferRef o, int oO,
                      int n, int iS) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineSwiGLU_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)iG offset:oG atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)iV offset:oV atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)o offset:oO atIndex:2];
  int total = n * iS;
  [enc dispatchThreads:MTLSizeMake(total, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(total, 256), 1, 1)];
  [mc barrier];
}

void Metal_Softmax_F16(MetalContextRef ctx, MetalBufferRef i, int oI,
                       MetalBufferRef r, int oR, int rs, int cs) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineSoftmax_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)i offset:oI atIndex:0];
  [enc setBytes:&cs length:4 atIndex:1];
  [enc setBytes:&rs length:4 atIndex:2];
  [enc dispatchThreads:MTLSizeMake(rs, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(rs, 256), 1, 1)];
  [mc barrier];
}

void Metal_StoreKV_F16(MetalContextRef ctx, MetalBufferRef k, int oK,
                       MetalBufferRef v, int oV, MetalBufferRef kC, int oKC,
                       MetalBufferRef vC, int oVC, int p, int h, int hd,
                       int windowSize) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineStoreKV_F16];
  int kv_dim = h * hd;
  [enc setBuffer:(__bridge id<MTLBuffer>)k offset:oK atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)v offset:oV atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)kC offset:oKC atIndex:2];
  [enc setBuffer:(__bridge id<MTLBuffer>)vC offset:oVC atIndex:3];
  [enc setBytes:&p length:4 atIndex:4];
  [enc setBytes:&kv_dim length:4 atIndex:5];
  [enc setBytes:&windowSize length:4 atIndex:6]; // Add window size
  [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(kv_dim, 256), 1, 1)];
  [mc barrier];
}

// Granular Attention Steps for Debugging
void Metal_AttScores_F16(MetalContextRef ctx, MetalBufferRef q, int oQ,
                         MetalBufferRef kC, int oK, MetalBufferRef s, int oS,
                         int p, int nh, int kh, int hd, int ctxLen,
                         int windowSize) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  int stride = ctxLen;
  [enc setComputePipelineState:mc.pipelineAttScores_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)q offset:oQ atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)kC offset:oK atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)s offset:oS atIndex:2];
  [enc setBytes:&p length:4 atIndex:3];
  [enc setBytes:&nh length:4 atIndex:4];
  [enc setBytes:&kh length:4 atIndex:5];
  [enc setBytes:&hd length:4 atIndex:6];
  [enc setBytes:&stride length:4 atIndex:7];
  [enc setBytes:&windowSize length:4 atIndex:8];
  [enc dispatchThreads:MTLSizeMake(nh * 32, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [mc barrier];
}

void Metal_AttSoftmax_F16(MetalContextRef ctx, MetalBufferRef s, int oS, int p,
                          int nh, int ctxLen) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  int stride = ctxLen;
  [enc setComputePipelineState:mc.pipelineSoftmax_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)s offset:oS atIndex:0];
  [enc setBytes:&p length:4 atIndex:1];
  [enc setBytes:&stride length:4 atIndex:2];
  [enc dispatchThreadgroups:MTLSizeMake(nh, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [mc barrier];
}

void Metal_AttValues_F16(MetalContextRef ctx, MetalBufferRef s, int oS,
                         MetalBufferRef vC, int oV, MetalBufferRef r, int oR,
                         int p, int nh, int kh, int hd, int ctxLen,
                         int windowSize) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  int stride = ctxLen;
  [enc setComputePipelineState:mc.pipelineAttValues_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)s offset:oS atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)vC offset:oV atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)r offset:oR atIndex:2];
  [enc setBytes:&p length:4 atIndex:3];
  [enc setBytes:&nh length:4 atIndex:4];
  [enc setBytes:&kh length:4 atIndex:5];
  [enc setBytes:&hd length:4 atIndex:6];
  [enc setBytes:&stride length:4 atIndex:7];
  [enc setBytes:&windowSize length:4 atIndex:8];
  int dim = nh * hd;
  [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(dim, 256), 1, 1)];
  [mc barrier];
}

void Metal_Attention_F16(MetalContextRef ctx, MetalBufferRef q, int oQ,
                         MetalBufferRef kC, int oK, MetalBufferRef vC, int oV,
                         MetalBufferRef r, int oR, MetalBufferRef s, int oS,
                         int p, int nh, int kh, int hd, int ctxLen,
                         int windowSize) {
  Metal_AttScores_F16(ctx, q, oQ, kC, oK, s, oS, p, nh, kh, hd, ctxLen,
                      windowSize);
  Metal_AttSoftmax_F16(ctx, s, oS, p, nh, ctxLen);
  Metal_AttValues_F16(ctx, s, oS, vC, oV, r, oR, p, nh, kh, hd, ctxLen,
                      windowSize);
}

void Metal_CopyBufferToF32(MetalContextRef ctx, MetalBufferRef src, void *dst,
                           int count) {
  id<MTLBuffer> buf = (__bridge id<MTLBuffer>)src;
  memcpy(dst, [buf contents], count * sizeof(float));
}

void Metal_RMSNormLinear_F16(MetalContextRef ctx, MetalBufferRef i, int oI,
                             MetalBufferRef nW, int oNW, MetalBufferRef w,
                             int oW, MetalBufferRef r, int oR, int iD, int oD,
                             float e, int batchSize) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineRMSNormLinear_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)i offset:oI atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)nW offset:oNW atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)w offset:oW atIndex:2];
  [enc setBuffer:(__bridge id<MTLBuffer>)r offset:oR atIndex:3];
  [enc setBytes:&e length:4 atIndex:4];
  [enc setBytes:&iD length:4 atIndex:5];
  [enc setBytes:&oD length:4 atIndex:6];
  [enc setBytes:&batchSize length:4 atIndex:7];
  [enc dispatchThreads:MTLSizeMake(32, oD, batchSize)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
}

void Metal_RMSNormLinear_Q4K_F16(MetalContextRef ctx, MetalBufferRef input,
                                 int offIn, MetalBufferRef normWeight,
                                 int offNormWeight, MetalBufferRef weight,
                                 int offWeight, MetalBufferRef result,
                                 int offRes, int M, int N, int K, float eps,
                                 float scale, int batchSize) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineRMSNormLinear_Q4K_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:offIn atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)normWeight
          offset:offNormWeight
         atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:offWeight atIndex:2];
  [enc setBuffer:(__bridge id<MTLBuffer>)result offset:offRes atIndex:3];
  [enc setBytes:&eps length:4 atIndex:4];
  [enc setBytes:&K length:4 atIndex:5];
  [enc setBytes:&N length:4 atIndex:6];
  [enc setBytes:&scale length:4 atIndex:7];
  [enc setBytes:&batchSize length:4 atIndex:8];

  [enc dispatchThreads:MTLSizeMake(1024, N, batchSize)
      threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  [mc barrier];
}

void Metal_SwiGLULinear_Q4K_F16(MetalContextRef ctx, MetalBufferRef gateIn,
                                int offGate, MetalBufferRef upIn, int offUp,
                                MetalBufferRef weight, int offWeight,
                                MetalBufferRef result, int offRes, int M, int N,
                                int K, float scale) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineSwiGLULinear_Q4K_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)gateIn offset:offGate atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)upIn offset:offUp atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:offWeight atIndex:2];
  [enc setBuffer:(__bridge id<MTLBuffer>)result offset:offRes atIndex:3];
  [enc setBytes:&K length:4 atIndex:4];
  [enc setBytes:&N length:4 atIndex:5];
  [enc setBytes:&scale length:4 atIndex:6];

  [enc dispatchThreads:MTLSizeMake(32, N, M)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
}

void Metal_MambaConv1d_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                           MetalBufferRef weight, int offW, MetalBufferRef bias,
                           int offB, MetalBufferRef state, int offS,
                           MetalBufferRef output, int offOut, int dim,
                           int kernelSize) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineMambaConv1d_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:offIn atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:offW atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)bias offset:offB atIndex:2];
  [enc setBuffer:(__bridge id<MTLBuffer>)state offset:offS atIndex:3];
  [enc setBuffer:(__bridge id<MTLBuffer>)output offset:offOut atIndex:4];
  [enc setBytes:&dim length:4 atIndex:5];
  [enc setBytes:&kernelSize length:4 atIndex:6];

  [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(dim, 256), 1, 1)];
  [mc barrier];
}

void Metal_MambaScan_F16(MetalContextRef ctx_ref, MetalBufferRef u, int offU,
                         MetalBufferRef h, int offH, MetalBufferRef A, int offA,
                         MetalBufferRef B, int offB, MetalBufferRef C, int offC,
                         MetalBufferRef D, int offD, MetalBufferRef dt,
                         int offDt, MetalBufferRef y, int offY, int dim,
                         int d_state) {
  @autoreleasepool {
    MetalWrapper *wrapper = (__bridge MetalWrapper *)ctx_ref;
    id<MTLComputeCommandEncoder> encoder = [wrapper ensureEncoder];
    [encoder setComputePipelineState:wrapper.pipelineMambaScan_F16];
    [encoder setBuffer:(__bridge id<MTLBuffer>)u offset:offU atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)h offset:offH atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:offA atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:offB atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)C offset:offC atIndex:4];
    [encoder setBuffer:(__bridge id<MTLBuffer>)D offset:offD atIndex:5];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dt offset:offDt atIndex:6];
    [encoder setBuffer:(__bridge id<MTLBuffer>)y offset:offY atIndex:7];

    int n_heads = 64;
    int head_dim = dim / n_heads;

    [encoder setBytes:&n_heads length:sizeof(int) atIndex:8];
    [encoder setBytes:&d_state length:sizeof(int) atIndex:9];
    [encoder setBytes:&head_dim length:sizeof(int) atIndex:10];

    MTLSize gridSize = MTLSizeMake(dim, 1, 1);
    NSUInteger threadGroupSize =
        wrapper.pipelineMambaScan_F16.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > dim)
      threadGroupSize = dim;
    MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];
    [wrapper barrier];
  }
}

void Metal_BatchedMatMul_F16(MetalContextRef ctx, MetalBufferRef a, int oA,
                             int sA, bool tA, MetalBufferRef b, int oB, int sB,
                             bool tB, MetalBufferRef c, int oC, int sC, int M,
                             int N, int K, int bC) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc endEncoding]; // MPS needs its own encoder (Blit/Compute) usually, or
                     // we use command buffer directly.
  mc.currentEncoder = nil;

  MPSMatrixDescriptor *descA =
      [MPSMatrixDescriptor matrixDescriptorWithRows:tA ? K : M
                                            columns:tA ? M : K
                                           rowBytes:sA
                                           dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *descB =
      [MPSMatrixDescriptor matrixDescriptorWithRows:tB ? N : K
                                            columns:tB ? K : N
                                           rowBytes:sB
                                           dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *descC =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                            columns:N
                                           rowBytes:sC
                                           dataType:MPSDataTypeFloat16];

  MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a
                                               offset:oA
                                           descriptor:descA];
  MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b
                                               offset:oB
                                           descriptor:descB];
  MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)c
                                               offset:oC
                                           descriptor:descC];

  MPSMatrixMultiplication *mul =
      [[MPSMatrixMultiplication alloc] initWithDevice:mc.device
                                        transposeLeft:tA
                                       transposeRight:tB
                                           resultRows:M
                                        resultColumns:N
                                      interiorColumns:K
                                                alpha:1.0
                                                 beta:0.0];

  [mul encodeToCommandBuffer:mc.currentCommandBuffer
                  leftMatrix:matA
                 rightMatrix:matB
                resultMatrix:matC];
}
void Metal_RMSNormQKV_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                          MetalBufferRef normWeight, int offNormWeight,
                          MetalBufferRef qWeight, int offQW,
                          MetalBufferRef kWeight, int offKW,
                          MetalBufferRef vWeight, int offVW,
                          MetalBufferRef qOut, int offQO, MetalBufferRef kOut,
                          int offKO, MetalBufferRef vOut, int offVO, int inDim,
                          int qDim, int kvDim, float eps, int batchSize) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineRMSNormQKV_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:offIn atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)normWeight
          offset:offNormWeight
         atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)qWeight offset:offQW atIndex:2];
  [enc setBuffer:(__bridge id<MTLBuffer>)kWeight offset:offKW atIndex:3];
  [enc setBuffer:(__bridge id<MTLBuffer>)vWeight offset:offVW atIndex:4];
  [enc setBuffer:(__bridge id<MTLBuffer>)qOut offset:offQO atIndex:5];
  [enc setBuffer:(__bridge id<MTLBuffer>)kOut offset:offKO atIndex:6];
  [enc setBuffer:(__bridge id<MTLBuffer>)vOut offset:offVO atIndex:7];
  [enc setBytes:&inDim length:4 atIndex:8];
  [enc setBytes:&qDim length:4 atIndex:9];
  [enc setBytes:&kvDim length:4 atIndex:10];
  [enc setBytes:&eps length:4 atIndex:11];
  [enc setBytes:&batchSize length:4 atIndex:12];

  int max_out = (qDim > kvDim) ? qDim : kvDim;
  [enc dispatchThreads:MTLSizeMake(32, max_out, batchSize)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
}
void Metal_FusedFFN_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                        MetalBufferRef normWeight, int offNormWeight,
                        MetalBufferRef gateWeight, int offGW,
                        MetalBufferRef upWeight, int offUW,
                        MetalBufferRef downWeight, int offDW,
                        MetalBufferRef output, int offOut, int inDim,
                        int interDim, float eps, int batchSize) {
  fprintf(stderr, "ERROR: Metal_FusedFFN_F16 is a stub! Update Go code to use "
                  "separate calls.\n");
}

// ==========================================
// FP32 Wrappers
// ==========================================

void Metal_RMSNorm_F32(MetalContextRef ctx, MetalBufferRef input, int offIn,
                       MetalBufferRef weight, int offWeight,
                       MetalBufferRef result, int offRes, int rows, int cols,
                       float eps) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineRMSNorm_F32];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:offIn atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)result offset:offRes atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:offWeight atIndex:2];
  [enc setBytes:&eps length:4 atIndex:3];
  [enc setBytes:&cols length:4 atIndex:4];
  int threads = (cols < 1024) ? cols : 1024;
  [enc dispatchThreads:MTLSizeMake(threads, rows, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [mc barrier];
}

void Metal_Add_F32(MetalContextRef ctx, MetalBufferRef a, int oA,
                   MetalBufferRef b, int oB, MetalBufferRef r, int oR,
                   int count) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineAdd_F32];
  [enc setBuffer:(__bridge id<MTLBuffer>)a offset:oA atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)b offset:oB atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)r offset:oR atIndex:2];
  [enc dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(count, 256), 1, 1)];
  [mc barrier];
}

void Metal_MatMul_Q4K_F32(MetalContextRef ctx, MetalBufferRef a, int offA,
                          int transA, MetalBufferRef b, int offB, int transB,
                          MetalBufferRef c, int offC, int M, int N, int K,
                          float scale) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];

  // Check if pipeline is valid
  if (!mc.pipelineMatMul_Q4K_F32) {
    fprintf(
        stderr,
        "ERROR: pipelineMatMul_Q4K_F32 is NULL! Kernel failed to compile.\n");
    fflush(stderr);
    return;
  }

  [enc setComputePipelineState:mc.pipelineMatMul_Q4K_F32];

  [enc setBuffer:(__bridge id<MTLBuffer>)a
          offset:offA
         atIndex:0]; // Weights (Q4K)
  [enc setBuffer:(__bridge id<MTLBuffer>)b
          offset:offB
         atIndex:1]; // Input (F32)
  [enc setBuffer:(__bridge id<MTLBuffer>)c
          offset:offC
         atIndex:2]; // Output (F32)
  [enc setBytes:&K length:4 atIndex:3];
  [enc setBytes:&N length:4 atIndex:4];
  [enc setBytes:&scale length:4 atIndex:5];

  [enc dispatchThreads:MTLSizeMake(32, N, M)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
}

void Metal_MatMul_Q6K_F32(MetalContextRef ctx, MetalBufferRef a, int offA,
                          int transA, MetalBufferRef b, int offB, int transB,
                          MetalBufferRef c, int offC, int M, int N, int K,
                          float scale) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];

  if (!mc.pipelineMatMul_Q6K_F32) {
    fprintf(
        stderr,
        "ERROR: pipelineMatMul_Q6K_F32 is NULL! Kernel failed to compile.\n");
    fflush(stderr);
    return;
  }

  [enc setComputePipelineState:mc.pipelineMatMul_Q6K_F32];

  [enc setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0]; // Weights
  [enc setBuffer:(__bridge id<MTLBuffer>)b offset:offB atIndex:1]; // Input
  [enc setBuffer:(__bridge id<MTLBuffer>)c offset:offC atIndex:2]; // Output
  [enc setBytes:&K length:4 atIndex:3];
  [enc setBytes:&N length:4 atIndex:4];
  [enc setBytes:&scale length:4 atIndex:5];

  [enc dispatchThreads:MTLSizeMake(32, N, M)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
}

void Metal_MatMul_Q4K_F32_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                              MetalBufferRef b, int offB, MetalBufferRef c,
                              int offC, int M, int N, int K, float scale) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineMatMul_Q4K_F32_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0]; // Weights
  [enc setBuffer:(__bridge id<MTLBuffer>)b offset:offB atIndex:1]; // Input
  [enc setBuffer:(__bridge id<MTLBuffer>)c offset:offC atIndex:2]; // Output
  [enc setBytes:&K length:4 atIndex:3];
  [enc setBytes:&N length:4 atIndex:4];
  [enc setBytes:&scale length:4 atIndex:5];
  [enc dispatchThreads:MTLSizeMake(32, N, M)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
}

void Metal_MatMul_Q6K_F32_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                              MetalBufferRef b, int offB, MetalBufferRef c,
                              int offC, int M, int N, int K, float scale) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineMatMul_Q6K_F32_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0]; // Weights
  [enc setBuffer:(__bridge id<MTLBuffer>)b offset:offB atIndex:1]; // Input
  [enc setBuffer:(__bridge id<MTLBuffer>)c offset:offC atIndex:2]; // Output
  [enc setBytes:&K length:4 atIndex:3];
  [enc setBytes:&N length:4 atIndex:4];
  [enc setBytes:&scale length:4 atIndex:5];
  [enc dispatchThreads:MTLSizeMake(32, N, M)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
}

void Metal_SwiGLU_F32(MetalContextRef ctx, MetalBufferRef iV, int oV,
                      MetalBufferRef iG, int oG, MetalBufferRef o, int oO,
                      int n, int iS) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineSwiGLU_F32];
  [enc setBuffer:(__bridge id<MTLBuffer>)iG offset:oG atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)iV offset:oV atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)o offset:oO atIndex:2];
  int total = n * iS;
  [enc dispatchThreads:MTLSizeMake(total, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(total, 256), 1, 1)];
  [mc barrier];
}

void Metal_Copy_F16(MetalContextRef ctx, MetalBufferRef src, int oS,
                    MetalBufferRef dst, int oD, int count) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineCopy_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)src offset:oS atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:oD atIndex:1];
  [enc dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
}

void Metal_Copy_F16_F32(MetalContextRef ctx, MetalBufferRef src, int oS,
                        MetalBufferRef dst, int oD, int n) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineCopy_F16_F32];
  [enc setBuffer:(__bridge id<MTLBuffer>)src offset:oS atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:oD atIndex:1];
  [enc dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(n, 256), 1, 1)];
  [mc barrier];
}

void Metal_Copy_F32_F16(MetalContextRef ctx, MetalBufferRef src, int oS,
                        MetalBufferRef dst, int oD, int n) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineCopy_F32_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)src offset:oS atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:oD atIndex:1];
  [enc dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(n, 256), 1, 1)];
  [mc barrier];
}

// Weights: F16, Input: F16, Output: F32
void Metal_MatMul_F16_F16_F32(MetalContextRef ctx, MetalBufferRef a, int offA,
                              MetalBufferRef b, int offB, MetalBufferRef c,
                              int offC, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:
           mc.pipelineMatMul_F16_F16_F32]; // Corrected:
                                           // linear_f16_in_f16_out_f32
  [enc setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)b offset:offB atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)c offset:offC atIndex:2];
  [enc setBytes:&K length:4 atIndex:3];
  [enc setBytes:&N length:4 atIndex:4];
  [enc dispatchThreads:MTLSizeMake(32, N, M)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
}

// Weights: F16, Input: F32, Output: F32
void Metal_MatMul_F16_F32_F32(MetalContextRef ctx, MetalBufferRef a, int offA,
                              MetalBufferRef b, int offB, MetalBufferRef c,
                              int offC, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineMatMul_F16_F32]; // linear_f16_f32
                                                           // (F32 input)
  [enc setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)b offset:offB atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)c offset:offC atIndex:2];
  [enc setBytes:&K length:4 atIndex:3];
  [enc setBytes:&N length:4 atIndex:4];
  [enc dispatchThreads:MTLSizeMake(32, N, M)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
}

void Metal_AttFused_F16(MetalContextRef ctx, MetalBufferRef q, int oQ,
                        MetalBufferRef kC, int oK, MetalBufferRef vC, int oV,
                        MetalBufferRef r, int oR, int p, int nh, int kh, int hd,
                        int windowSize, int maxCtxLen) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineAttFused_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)q offset:oQ atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)kC offset:oK atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)vC offset:oV atIndex:2];
  [enc setBuffer:(__bridge id<MTLBuffer>)r offset:oR atIndex:3];
  [enc setBytes:&p length:4 atIndex:4];
  [enc setBytes:&nh length:4 atIndex:5];
  [enc setBytes:&kh length:4 atIndex:6];
  [enc setBytes:&hd length:4 atIndex:7];
  [enc setBytes:&windowSize length:4 atIndex:8];
  [enc setBytes:&maxCtxLen length:4 atIndex:9];
  [enc dispatchThreadgroups:MTLSizeMake(nh, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  [mc barrier];
}

// FP32 FFN Bridge Functions for Small Models

void Metal_LinearF16ToF32(MetalContextRef ctx, MetalBufferRef weight,
                          int offWeight, MetalBufferRef input, int offInput,
                          MetalBufferRef output, int offOutput, int rows,
                          int dimIn, int dimOut) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineLinearF16ToF32];
  [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:offWeight atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:offInput atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)output offset:offOutput atIndex:2];
  [enc setBytes:&dimIn length:4 atIndex:3];
  [enc setBytes:&dimOut length:4 atIndex:4];
  [enc dispatchThreadgroups:MTLSizeMake(1, dimOut, rows)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [mc barrier];
}

void Metal_LinearF32ToF16(MetalContextRef ctx, MetalBufferRef weight,
                          int offWeight, MetalBufferRef input, int offInput,
                          MetalBufferRef output, int offOutput, int rows,
                          int dimIn, int dimOut) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineLinearF32ToF16];
  [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:offWeight atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:offInput atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)output offset:offOutput atIndex:2];
  [enc setBytes:&dimIn length:4 atIndex:3];
  [enc setBytes:&dimOut length:4 atIndex:4];
  [enc dispatchThreadgroups:MTLSizeMake(1, dimOut, rows)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [mc barrier];
}

void Metal_RMSNorm_F32_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                           MetalBufferRef weight, int offWeight,
                           MetalBufferRef result, int offRes, int rows,
                           int cols, float eps) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineRMSNorm_F32_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:offIn atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)result offset:offRes atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:offWeight atIndex:2];
  [enc setBytes:&eps length:4 atIndex:3];
  [enc setBytes:&cols length:4 atIndex:4];
  int threads = (cols < 1024) ? cols : 1024;
  [enc dispatchThreads:MTLSizeMake(threads, rows, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [mc barrier];
}

void Metal_Add_F32_F16(MetalContextRef ctx, MetalBufferRef a, int oA,
                       MetalBufferRef b, int oB, MetalBufferRef r, int oR,
                       int count) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineAdd_F32_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)a
          offset:oA
         atIndex:0]; // FP32 accumulator
  [enc setBuffer:(__bridge id<MTLBuffer>)b offset:oB atIndex:1]; // FP16 delta
  [enc setBuffer:(__bridge id<MTLBuffer>)r offset:oR atIndex:2]; // FP32 result
  [enc dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(count, 256), 1, 1)];
  [mc barrier];
}

void Metal_LinearQ6K_F16_F32(MetalContextRef ctx, MetalBufferRef weight,
                             int offWeight, MetalBufferRef input, int offInput,
                             MetalBufferRef output, int offOutput, int rows,
                             int dimIn, int dimOut, float scale) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineLinearQ6K_F16_F32];
  [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:offWeight atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:offInput atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)output offset:offOutput atIndex:2];
  [enc setBytes:&dimIn length:4 atIndex:3];
  [enc setBytes:&dimOut length:4 atIndex:4];
  [enc setBytes:&scale length:4 atIndex:5];
  [enc dispatchThreadgroups:MTLSizeMake(1, dimOut, rows)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [mc barrier];
}

void Metal_LinearQ4_0_F16(MetalContextRef ctx, MetalBufferRef weight,
                          int offWeight, MetalBufferRef input, int offInput,
                          MetalBufferRef output, int offOutput, int rows,
                          int dimIn, int dimOut, float scale) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineLinearQ4_0_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:offWeight atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:offInput atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)output offset:offOutput atIndex:2];
  [enc setBytes:&dimIn length:4 atIndex:3];
  [enc setBytes:&dimOut length:4 atIndex:4];
  [enc setBytes:&scale length:4 atIndex:5];
  [enc dispatchThreadgroups:MTLSizeMake(1, dimOut, rows)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [mc barrier];
}

void Metal_LinearQ4_0_F32(MetalContextRef ctx, MetalBufferRef weight,
                          int offWeight, MetalBufferRef input, int offInput,
                          MetalBufferRef output, int offOutput, int rows,
                          int dimIn, int dimOut, float scale) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineLinearQ4_0_F32];
  [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:offWeight atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:offInput atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)output offset:offOutput atIndex:2];
  [enc setBytes:&dimIn length:4 atIndex:3];
  [enc setBytes:&dimOut length:4 atIndex:4];
  [enc setBytes:&scale length:4 atIndex:5];
  [enc dispatchThreadgroups:MTLSizeMake(1, dimOut, rows)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [mc barrier];
}

void Metal_EmbeddingQ4_0_F16(MetalContextRef ctx, MetalBufferRef weights,
                             int offW, MetalBufferRef result, int offRes,
                             int rowIdx, int cols) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineEmbeddingQ4_0_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)weights offset:offW atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)result offset:offRes atIndex:1];
  [enc setBytes:&rowIdx length:4 atIndex:2];
  [enc setBytes:&cols length:4 atIndex:3];
  int threads = (cols < 1024) ? cols : 1024;
  [enc dispatchThreads:MTLSizeMake(cols, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [mc barrier];
}

void Metal_StoreKV_F16_Batch(MetalContextRef ctx, MetalBufferRef k, int offK,
                             MetalBufferRef v, int offV, MetalBufferRef kC,
                             int offKC, MetalBufferRef vC, int offVC, int p,
                             int h, int hd, int windowSize, int batchSize) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineStoreKV_F16_Batch];
  int kv_dim = h * hd;
  [enc setBuffer:(__bridge id<MTLBuffer>)k offset:offK atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)v offset:offV atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)kC offset:offKC atIndex:2];
  [enc setBuffer:(__bridge id<MTLBuffer>)vC offset:offVC atIndex:3];
  [enc setBytes:&p length:4 atIndex:4];
  [enc setBytes:&kv_dim length:4 atIndex:5];
  [enc setBytes:&windowSize length:4 atIndex:6];

  [enc dispatchThreads:MTLSizeMake(kv_dim, batchSize, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(kv_dim, 256), 1, 1)];
  [mc barrier];
}

void Metal_RMSNormQKV_Q4K_F16(
    MetalContextRef ctx, MetalBufferRef input, int offIn,
    MetalBufferRef normWeight, int offNormWeight, MetalBufferRef qWeight,
    int offQW, MetalBufferRef kWeight, int offKW, MetalBufferRef vWeight,
    int offVW, MetalBufferRef qOut, int offQO, MetalBufferRef kOut, int offKO,
    MetalBufferRef vOut, int offVO, int inDim, int qDim, int kvDim, float eps,
    float scale, int batchSize) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineRMSNormQKV_Q4K_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:offIn atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)normWeight
          offset:offNormWeight
         atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)qWeight offset:offQW atIndex:2];
  [enc setBuffer:(__bridge id<MTLBuffer>)kWeight offset:offKW atIndex:3];
  [enc setBuffer:(__bridge id<MTLBuffer>)vWeight offset:offVW atIndex:4];
  [enc setBuffer:(__bridge id<MTLBuffer>)qOut offset:offQO atIndex:5];
  [enc setBuffer:(__bridge id<MTLBuffer>)kOut offset:offKO atIndex:6];
  [enc setBuffer:(__bridge id<MTLBuffer>)vOut offset:offVO atIndex:7];
  [enc setBytes:&inDim length:4 atIndex:8];
  [enc setBytes:&qDim length:4 atIndex:9];
  [enc setBytes:&kvDim length:4 atIndex:10];
  [enc setBytes:&eps length:4 atIndex:11];
  [enc setBytes:&scale length:4 atIndex:12];
  [enc setBytes:&batchSize length:4 atIndex:13];

  int max_out_dim = qDim > kvDim ? qDim : kvDim;
  [enc dispatchThreads:MTLSizeMake(32, max_out_dim, batchSize)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
}

void Metal_AttPaged_F16(MetalContextRef ctx, MetalBufferRef q, int offQ,
                        MetalBufferRef kC, int offK, MetalBufferRef vC,
                        int offV, MetalBufferRef r, int offR,
                        MetalBufferRef blockTable, int offBT, int p, int nh,
                        int kh, int hd, int blockSize, int maxCtxLen) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineAttPaged_F16];

  [enc setBuffer:(__bridge id<MTLBuffer>)q offset:offQ atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)kC offset:offK atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)vC offset:offV atIndex:2];
  [enc setBuffer:(__bridge id<MTLBuffer>)r offset:offR atIndex:3];

  [enc setBytes:&p length:sizeof(int) atIndex:4];
  [enc setBytes:&nh length:sizeof(int) atIndex:5];
  [enc setBytes:&kh length:sizeof(int) atIndex:6];
  [enc setBytes:&hd length:sizeof(int) atIndex:7];
  [enc setBytes:&blockSize length:sizeof(int) atIndex:8];
  [enc setBytes:&maxCtxLen length:sizeof(int) atIndex:9];

  [enc setBuffer:(__bridge id<MTLBuffer>)blockTable offset:offBT atIndex:10];

  MTLSize threadGroupSize = MTLSizeMake(128, 1, 1);
  MTLSize gridSize = MTLSizeMake(nh * 128, 1, 1);

  [enc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
  [mc barrier];
}

void Metal_SiLU_F16(MetalContextRef ctx_ref, MetalBufferRef input, int offIn,
                    MetalBufferRef output, int offOut, int n) {
  @autoreleasepool {
    MetalWrapper *wrapper = (__bridge MetalWrapper *)ctx_ref;
    id<MTLComputeCommandEncoder> encoder = [wrapper ensureEncoder];
    [encoder setComputePipelineState:wrapper.pipelineSiLU_F16];
    [encoder setBuffer:(__bridge id<MTLBuffer>)input offset:offIn atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)output offset:offOut atIndex:1];

    MTLSize gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger threadGroupSize =
        wrapper.pipelineSiLU_F16.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > n)
      threadGroupSize = n;
    MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];
  }
}

void Metal_Slice_F16(MetalContextRef ctx_ref, MetalBufferRef input, int offIn,
                     MetalBufferRef output, int offOut, int startCol,
                     int numCols, int totalCols, int rows) {
  @autoreleasepool {
    MetalWrapper *wrapper = (__bridge MetalWrapper *)ctx_ref;
    id<MTLComputeCommandEncoder> encoder = [wrapper ensureEncoder];
    [encoder setComputePipelineState:wrapper.pipelineSlice_F16];
    [encoder setBuffer:(__bridge id<MTLBuffer>)input offset:offIn atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)output offset:offOut atIndex:1];
    [encoder setBytes:&startCol length:sizeof(int) atIndex:2];
    [encoder setBytes:&numCols length:sizeof(int) atIndex:3];
    [encoder setBytes:&totalCols length:sizeof(int) atIndex:4];

    int n = rows * numCols;
    MTLSize gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger threadGroupSize =
        wrapper.pipelineSlice_F16.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > n)
      threadGroupSize = n;
    MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];
  }
}

void Metal_Mul_F16(MetalContextRef ctx_ref, MetalBufferRef a, int offA,
                   MetalBufferRef b, int offB, MetalBufferRef result,
                   int offRes, int n) {
  @autoreleasepool {
    MetalWrapper *wrapper = (__bridge MetalWrapper *)ctx_ref;
    id<MTLComputeCommandEncoder> encoder = [wrapper ensureEncoder];
    [encoder setComputePipelineState:wrapper.pipelineMul_F16];
    [encoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)b offset:offB atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)result offset:offRes atIndex:2];

    MTLSize gridSize = MTLSizeMake(n, 1, 1);
    NSUInteger threadGroupSize =
        wrapper.pipelineMul_F16.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > n)
      threadGroupSize = n;
    MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];
  }
}
