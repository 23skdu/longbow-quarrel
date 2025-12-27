#import "metal_bridge.h"
#import <Metal/Metal.h>

@interface MetalWrapper : NSObject
@property(strong) id<MTLDevice> device;
@property(strong) id<MTLCommandQueue> commandQueue;
@property(strong) id<MTLLibrary> library;
@property(strong) id<MTLComputePipelineState> pipelineRMSNorm_F16;
@property(strong) id<MTLComputePipelineState> pipelineMatMul_F16;
@property(strong) id<MTLComputePipelineState> pipelineRoPE_F16;
@property(strong) id<MTLComputePipelineState> pipelineEmbedding_F16;
@property(strong) id<MTLComputePipelineState> pipelineAdd_F16;
@property(strong) id<MTLComputePipelineState> pipelineCopy_F16;
@property(strong) id<MTLComputePipelineState> pipelineSwiGLU_F16;
@property(strong) id<MTLComputePipelineState> pipelineSoftmax_F16;
@property(strong) id<MTLComputePipelineState> pipelineAttScores_F16;
@property(strong) id<MTLComputePipelineState> pipelineAttValues_F16;
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
  if (!f)
    return nil;
  return [ctx.device newComputePipelineStateWithFunction:f error:nil];
}

MetalContextRef Metal_Init(const char *libSource) {
  MetalWrapper *ctx = [[MetalWrapper alloc] init];
  ctx.device = MTLCreateSystemDefaultDevice();
  ctx.commandQueue = [ctx.device newCommandQueue];
  ctx.library =
      [ctx.device newLibraryWithSource:[NSString stringWithUTF8String:libSource]
                               options:nil
                                 error:nil];
  if (!ctx.library)
    return NULL;
  ctx.pipelineRMSNorm_F16 = loadPipeline(ctx, @"rmsnorm_f16");
  ctx.pipelineMatMul_F16 = loadPipeline(ctx, @"linear_f16");
  ctx.pipelineRoPE_F16 = loadPipeline(ctx, @"rope_f16");
  ctx.pipelineEmbedding_F16 = loadPipeline(ctx, @"embedding_f16");
  ctx.pipelineAdd_F16 = loadPipeline(ctx, @"add_f16");
  ctx.pipelineCopy_F16 = loadPipeline(ctx, @"copy_f16");
  ctx.pipelineSwiGLU_F16 = loadPipeline(ctx, @"swiglu_f16");
  ctx.pipelineSoftmax_F16 = loadPipeline(ctx, @"softmax_f16");
  ctx.pipelineAttScores_F16 = loadPipeline(ctx, @"att_scores_f16");
  ctx.pipelineAttValues_F16 = loadPipeline(ctx, @"att_values_f16");
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
  memcpy([(__bridge id<MTLBuffer>)buf contents] + o, d, s);
}
void Metal_CopyToHost(MetalBufferRef buf, int o, void *d, int s) {
  memcpy(d, [(__bridge id<MTLBuffer>)buf contents] + o, s);
}
void *Metal_GetBufferContents(MetalBufferRef buf) {
  return [(__bridge id<MTLBuffer>)buf contents];
}
void Metal_ZeroBuffer(MetalBufferRef buf, int o, int s) {
  memset([(__bridge id<MTLBuffer>)buf contents] + o, 0, s);
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
  [enc dispatchThreads:MTLSizeMake(cols, rows, 1)
      threadsPerThreadgroup:MTLSizeMake(cols, 1, 1)];
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
}

void Metal_Layer_F16(MetalContextRef ctx, MetalBufferRef input,
                     MetalBufferRef attnNormW, MetalBufferRef qW,
                     MetalBufferRef kW, MetalBufferRef vW, MetalBufferRef oW,
                     MetalBufferRef ffnNormW, MetalBufferRef ffnGateW,
                     MetalBufferRef ffnUpW, MetalBufferRef ffnDownW,
                     MetalBufferRef kCache, MetalBufferRef vCache,
                     MetalBufferRef scratch1, MetalBufferRef scratch2,
                     MetalBufferRef scratch3, MetalBufferRef scratch4, int pos,
                     int numHeads, int kvHeads, int headDim, int interDim,
                     float eps, float ropeTheta, int ctxLen) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  int dim = numHeads * headDim, kv_dim = kvHeads * headDim;

  // Attn Norm: input -> scratch4
  [enc setComputePipelineState:mc.pipelineRMSNorm_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch4 offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)attnNormW offset:0 atIndex:2];
  [enc setBytes:&eps length:4 atIndex:3];
  [enc setBytes:&dim length:4 atIndex:4];
  [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(dim, 256), 1, 1)];
  [mc barrier];

  // QKV Projects: read scratch4 (normed)
  [enc setComputePipelineState:mc.pipelineMatMul_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)qW offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch4 offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch1 offset:0 atIndex:2];
  [enc setBytes:&dim length:4 atIndex:3];
  [enc setBytes:&dim length:4 atIndex:4];
  [enc dispatchThreads:MTLSizeMake(32, dim, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [enc setBuffer:(__bridge id<MTLBuffer>)kW offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch2 offset:0 atIndex:2];
  [enc setBytes:&kv_dim length:4 atIndex:4];
  [enc dispatchThreads:MTLSizeMake(32, kv_dim, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [enc setBuffer:(__bridge id<MTLBuffer>)vW offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch3 offset:0 atIndex:2];
  [enc dispatchThreads:MTLSizeMake(32, kv_dim, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];

  [enc setComputePipelineState:mc.pipelineRoPE_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch1 offset:0 atIndex:0];
  [enc setBytes:&pos length:4 atIndex:1];
  [enc setBytes:&headDim length:4 atIndex:2];
  [enc setBytes:&ropeTheta length:4 atIndex:3];
  int q_rope = numHeads * (headDim / 2);
  [enc dispatchThreads:MTLSizeMake(q_rope, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(q_rope, 256), 1, 1)];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch2 offset:0 atIndex:0];
  int kv_rope = kvHeads * (headDim / 2);
  [enc dispatchThreads:MTLSizeMake(kv_rope, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(kv_rope, 256), 1, 1)];
  [mc barrier];

  [enc setComputePipelineState:mc.pipelineCopy_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch2 offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)kCache
          offset:pos * kv_dim * 2
         atIndex:1];
  [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(kv_dim, 256), 1, 1)];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch3 offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)vCache
          offset:pos * kv_dim * 2
         atIndex:1];
  [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(kv_dim, 256), 1, 1)];
  [mc barrier];

  [enc setComputePipelineState:mc.pipelineAttScores_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch1 offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)kCache offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch4 offset:0 atIndex:2];
  [enc setBytes:&pos length:4 atIndex:3];
  [enc setBytes:&numHeads length:4 atIndex:4];
  [enc setBytes:&kvHeads length:4 atIndex:5];
  [enc setBytes:&headDim length:4 atIndex:6];
  [enc setBytes:&ctxLen length:4 atIndex:7];
  [enc dispatchThreads:MTLSizeMake(numHeads * 32, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [mc barrier];
  [enc setComputePipelineState:mc.pipelineSoftmax_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch4 offset:0 atIndex:0];
  [enc setBytes:&pos length:4 atIndex:1];
  [enc setBytes:&ctxLen length:4 atIndex:2];
  [enc dispatchThreads:MTLSizeMake(numHeads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(numHeads, 1, 1)];
  [mc barrier];
  [enc setComputePipelineState:mc.pipelineAttValues_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch4 offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)vCache offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch1 offset:0 atIndex:2];
  [enc setBytes:&pos length:4 atIndex:3];
  [enc setBytes:&numHeads length:4 atIndex:4];
  [enc setBytes:&kvHeads length:4 atIndex:5];
  [enc setBytes:&headDim length:4 atIndex:6];
  [enc setBytes:&ctxLen length:4 atIndex:7];
  [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(dim, 256), 1, 1)];
  [mc barrier];

  [enc setComputePipelineState:mc.pipelineMatMul_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)oW offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch1 offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch2 offset:0 atIndex:2];
  [enc setBytes:&dim length:4 atIndex:3];
  [enc setBytes:&dim length:4 atIndex:4];
  [enc dispatchThreads:MTLSizeMake(32, dim, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
  [enc setComputePipelineState:mc.pipelineAdd_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch2 offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:2];
  [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(dim, 256), 1, 1)];
  [mc barrier];

  // FFN Norm: input -> scratch4
  [enc setComputePipelineState:mc.pipelineRMSNorm_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch4 offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)ffnNormW offset:0 atIndex:2];
  [enc setBytes:&eps length:4 atIndex:3];
  [enc setBytes:&dim length:4 atIndex:4];
  [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(dim, 1, 1)];
  [mc barrier];

  // FFN MatMuls read scratch4 (normed)
  [enc setComputePipelineState:mc.pipelineMatMul_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)ffnGateW offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch4 offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch1 offset:0 atIndex:2];
  [enc setBytes:&dim length:4 atIndex:3];
  [enc setBytes:&interDim length:4 atIndex:4];
  [enc dispatchThreads:MTLSizeMake(32, interDim, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [enc setBuffer:(__bridge id<MTLBuffer>)ffnUpW offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch2 offset:0 atIndex:2];
  [enc dispatchThreads:MTLSizeMake(32, interDim, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
  [enc setComputePipelineState:mc.pipelineSwiGLU_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch1 offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch2 offset:0 atIndex:1];
  [enc dispatchThreads:MTLSizeMake(interDim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(interDim, 256), 1, 1)];
  [mc barrier];
  [enc setComputePipelineState:mc.pipelineMatMul_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)ffnDownW offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch1 offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch2 offset:0 atIndex:2];
  [enc setBytes:&interDim length:4 atIndex:3];
  [enc setBytes:&dim length:4 atIndex:4];
  [enc dispatchThreads:MTLSizeMake(32, dim, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
  [mc barrier];
  [enc setComputePipelineState:mc.pipelineAdd_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)scratch2 offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:2];
  [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(dim, 256), 1, 1)];
}

void Metal_Scale_F16(MetalContextRef ctx, MetalBufferRef a, int oA, uint16_t v,
                     MetalBufferRef r, int oR, int n) {}
void Metal_RoPE_F16(MetalContextRef ctx, MetalBufferRef d, int oD, int b, int s,
                    int nh, int hd, int po, float rt) {}
void Metal_SwiGLU_F16(MetalContextRef ctx, MetalBufferRef iV, int oV,
                      MetalBufferRef iG, int oG, MetalBufferRef o, int oO,
                      int n, int iS) {}
void Metal_Softmax_F16(MetalContextRef ctx, MetalBufferRef i, int oI,
                       MetalBufferRef r, int oR, int rs, int cs) {}
void Metal_StoreKV_F16(MetalContextRef ctx, MetalBufferRef k, int oK,
                       MetalBufferRef v, int oV, MetalBufferRef kC,
                       MetalBufferRef vC, int p, int h, int hd) {}
void Metal_Attention_F16(MetalContextRef ctx, MetalBufferRef q, int oQ,
                         MetalBufferRef kC, MetalBufferRef vC, MetalBufferRef r,
                         int oR, int p, int nh, int kh, int hd) {}
void Metal_RMSNormLinear_F16(MetalContextRef ctx, MetalBufferRef i, int oI,
                             MetalBufferRef nW, int oNW, MetalBufferRef w,
                             int oW, MetalBufferRef r, int oR, int iD, int oD,
                             float e) {}
void Metal_BatchedMatMul_F16(MetalContextRef ctx, MetalBufferRef a, int oA,
                             int sA, bool tA, MetalBufferRef b, int oB, int sB,
                             bool tB, MetalBufferRef c, int oC, int sC, int M,
                             int N, int K, int bC) {}
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
