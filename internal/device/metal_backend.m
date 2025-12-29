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
@property(strong) id<MTLComputePipelineState> pipelineAdd_F16;
@property(strong) id<MTLComputePipelineState> pipelineCopy_F16;
@property(strong) id<MTLComputePipelineState> pipelineSwiGLU_F16;
@property(strong) id<MTLComputePipelineState> pipelineSoftmax_F16;
@property(strong) id<MTLComputePipelineState> pipelineAttScores_F16;
@property(strong) id<MTLComputePipelineState> pipelineAttValues_F16;
@property(strong) id<MTLComputePipelineState> pipelineAttFused_F16;
@property(strong) id<MTLComputePipelineState> pipelineScale_F16;
// FP32 Pipelines
@property(strong) id<MTLComputePipelineState> pipelineRMSNorm_F32;
@property(strong) id<MTLComputePipelineState> pipelineMatMul_Q4K_F32;
@property(strong) id<MTLComputePipelineState> pipelineAdd_F32;
@property(strong) id<MTLComputePipelineState> pipelineSwiGLU_F32;
@property(strong) id<MTLComputePipelineState> pipelineMatMul_F16_F32;
@property(strong) id<MTLComputePipelineState> pipelineCopy_F16_F32;
@property(strong) id<MTLComputePipelineState> pipelineCopy_F32_F16;
@property(strong) id<MTLCommandBuffer> currentCommandBuffer;
@property(strong) id<MTLComputeCommandEncoder> currentEncoder;
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
    [self.currentCommandBuffer commit];
    [self.currentCommandBuffer waitUntilCompleted];
    self.currentCommandBuffer = nil;
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
  ctx.pipelineAdd_F16 = loadPipeline(ctx, @"add_f16");
  ctx.pipelineCopy_F16 = loadPipeline(ctx, @"copy_f16");
  ctx.pipelineSwiGLU_F16 = loadPipeline(ctx, @"swiglu_f16");
  ctx.pipelineSoftmax_F16 = loadPipeline(ctx, @"softmax_f16");
  ctx.pipelineAttScores_F16 = loadPipeline(ctx, @"att_scores_f16");
  ctx.pipelineAttValues_F16 = loadPipeline(ctx, @"att_values_f16");
  ctx.pipelineAttFused_F16 = loadPipeline(ctx, @"att_fused_f16");
  ctx.pipelineScale_F16 = loadPipeline(ctx, @"scale_f16");

  // FP32 Load
  ctx.pipelineRMSNorm_F32 = loadPipeline(ctx, @"rmsnorm_f32");
  ctx.pipelineMatMul_Q4K_F32 = loadPipeline(ctx, @"linear_q4k_f32");
  ctx.pipelineAdd_F32 = loadPipeline(ctx, @"add_f32");
  ctx.pipelineSwiGLU_F32 = loadPipeline(ctx, @"swiglu_f32");
  ctx.pipelineMatMul_F16_F32 = loadPipeline(ctx, @"linear_f16_f32");
  ctx.pipelineCopy_F16_F32 = loadPipeline(ctx, @"copy_f16_to_f32");
  ctx.pipelineCopy_F32_F16 = loadPipeline(ctx, @"copy_f32_to_f16");

  return (__bridge_retained MetalContextRef)ctx;
}

void Metal_Free(MetalContextRef ctx) {
  MetalWrapper *mc = (__bridge_transfer MetalWrapper *)ctx;
  [mc flush];
}
void Metal_Synchronize(MetalContextRef ctx) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  [mc flush];
}
MetalBufferRef Metal_Alloc(MetalContextRef ctx, int size) {
  return (
      __bridge_retained MetalBufferRef)[[(__bridge MetalWrapper *)ctx device]
      newBufferWithLength:size
                  options:MTLResourceStorageModeShared];
}
void Metal_FreeBuffer(MetalContextRef ctx, MetalBufferRef buf) {
  id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)buf;
  b = nil;
}
void Metal_CopyToDevice(MetalBufferRef buf, int o, const void *d, int s) {
  id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf;
  memcpy([b contents] + o, d, s);
  [b didModifyRange:NSMakeRange(o, s)];
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
                          MetalBufferRef c, int offC, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineMatMul_Q4K_F16];

  [enc setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)b offset:offB atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)c offset:offC atIndex:2];
  [enc setBytes:&K length:4 atIndex:3];
  [enc setBytes:&N length:4 atIndex:4];

  [enc dispatchThreads:MTLSizeMake(32, N, M)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
}

void Metal_MatMul_Q3K_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                          bool transA, MetalBufferRef b, int offB, bool transB,
                          MetalBufferRef c, int offC, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineMatMul_Q3K_F16];

  [enc setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)b offset:offB atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)c offset:offC atIndex:2];
  [enc setBytes:&K length:4 atIndex:3];
  [enc setBytes:&N length:4 atIndex:4];

  [enc dispatchThreads:MTLSizeMake(32, N, M)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
}

void Metal_MatMul_Q6K_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                          bool transA, MetalBufferRef b, int offB, bool transB,
                          MetalBufferRef c, int offC, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineMatMul_Q6K_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)b offset:offB atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)c offset:offC atIndex:2];
  [enc setBytes:&K length:4 atIndex:3];
  [enc setBytes:&N length:4 atIndex:4];
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

  int pairs = nh * (hd / 2);
  [enc dispatchThreads:MTLSizeMake(pairs, 1, 1)
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
                       MetalBufferRef v, int oV, MetalBufferRef kC,
                       MetalBufferRef vC, int p, int h, int hd) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineCopy_F16];
  int kv_dim = h * hd;
  [enc setBuffer:(__bridge id<MTLBuffer>)k offset:oK atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)kC offset:p * kv_dim * 2 atIndex:1];
  [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(kv_dim, 256), 1, 1)];
  [enc setBuffer:(__bridge id<MTLBuffer>)v offset:oV atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)vC offset:p * kv_dim * 2 atIndex:1];
  [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(kv_dim, 256), 1, 1)];
  [mc barrier];
}

// Granular Attention Steps for Debugging
void Metal_AttScores_F16(MetalContextRef ctx, MetalBufferRef q, int oQ,
                         MetalBufferRef kC, MetalBufferRef s, int oS, int p,
                         int nh, int kh, int hd, int ctxLen) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  int stride = ctxLen;
  [enc setComputePipelineState:mc.pipelineAttScores_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)q offset:oQ atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)kC offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)s offset:oS atIndex:2];
  [enc setBytes:&p length:4 atIndex:3];
  [enc setBytes:&nh length:4 atIndex:4];
  [enc setBytes:&kh length:4 atIndex:5];
  [enc setBytes:&hd length:4 atIndex:6];
  [enc setBytes:&stride length:4 atIndex:7];
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
                         MetalBufferRef vC, MetalBufferRef r, int oR, int p,
                         int nh, int kh, int hd, int ctxLen) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  int stride = ctxLen;
  [enc setComputePipelineState:mc.pipelineAttValues_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)s offset:oS atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)vC offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)r offset:oR atIndex:2];
  [enc setBytes:&p length:4 atIndex:3];
  [enc setBytes:&nh length:4 atIndex:4];
  [enc setBytes:&kh length:4 atIndex:5];
  [enc setBytes:&hd length:4 atIndex:6];
  [enc setBytes:&stride length:4 atIndex:7];
  int dim = nh * hd;
  [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(dim, 256), 1, 1)];
  [mc barrier];
}

void Metal_Attention_F16(MetalContextRef ctx, MetalBufferRef q, int oQ,
                         MetalBufferRef kC, MetalBufferRef vC, MetalBufferRef r,
                         int oR, MetalBufferRef s, int oS, int p, int nh,
                         int kh, int hd, int ctxLen) {
  Metal_AttScores_F16(ctx, q, oQ, kC, s, oS, p, nh, kh, hd, ctxLen);
  Metal_AttSoftmax_F16(ctx, s, oS, p, nh, ctxLen);
  Metal_AttValues_F16(ctx, s, oS, vC, r, oR, p, nh, kh, hd, ctxLen);
}

void Metal_RMSNormLinear_F16(MetalContextRef ctx, MetalBufferRef i, int oI,
                             MetalBufferRef nW, int oNW, MetalBufferRef w,
                             int oW, MetalBufferRef r, int oR, int iD, int oD,
                             float e) {}
void Metal_BatchedMatMul_F16(MetalContextRef ctx, MetalBufferRef a, int oA,
                             int sA, bool tA, MetalBufferRef b, int oB, int sB,
                             bool tB, MetalBufferRef c, int oC, int sC, int M,
                             int N, int K, int bC) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc endEncoding]; // MPS needs its own encoder (Blit/Compute) usually, or we
                     // use command buffer directly.
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
                          int qDim, int kvDim, float eps) {}
void Metal_FusedFFN_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                        MetalBufferRef normWeight, int offNormWeight,
                        MetalBufferRef gateWeight, int offGW,
                        MetalBufferRef upWeight, int offUW,
                        MetalBufferRef downWeight, int offDW,
                        MetalBufferRef output, int offOut, int inDim,
                        int interDim, float eps) {}

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
                          MetalBufferRef c, int offC, int M, int N, int K) {
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

void Metal_MatMul_F16_F32(MetalContextRef ctx, MetalBufferRef a, int offA,
                          MetalBufferRef b, int offB, MetalBufferRef c,
                          int offC, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineMatMul_F16_F32];
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
                        MetalBufferRef kC, MetalBufferRef vC, MetalBufferRef r,
                        int oR, int p, int nh, int kh, int hd) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineAttFused_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)q offset:oQ atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)kC offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)vC offset:0 atIndex:2];
  [enc setBuffer:(__bridge id<MTLBuffer>)r offset:oR atIndex:3];
  [enc setBytes:&p length:4 atIndex:4];
  [enc setBytes:&nh length:4 atIndex:5];
  [enc setBytes:&kh length:4 atIndex:6];
  [enc setBytes:&hd length:4 atIndex:7];
  [enc dispatchThreadgroups:MTLSizeMake(nh, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  [mc barrier];
}
