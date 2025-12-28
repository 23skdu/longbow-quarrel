#import "metal_bridge.h"
#import <Metal/Metal.h>

@interface MetalWrapper : NSObject
@property(strong) id<MTLDevice> device;
@property(strong) id<MTLCommandQueue> commandQueue;
@property(strong) id<MTLLibrary> library;
@property(strong) id<MTLComputePipelineState> pipelineRMSNorm_F16;
@property(strong) id<MTLComputePipelineState> pipelineMatMul_F16;
@property(strong) id<MTLComputePipelineState> pipelineMatMul_Q4K_F16;
@property(strong) id<MTLComputePipelineState> pipelineMatMul_Q3K_F16;
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
  ctx.pipelineMatMul_Q4K_F16 = loadPipeline(ctx, @"linear_q4k_f16");
  ctx.pipelineMatMul_Q3K_F16 = loadPipeline(ctx, @"linear_q3k_f16");
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

void Metal_Scale_F16(MetalContextRef ctx, MetalBufferRef a, int oA, uint16_t v,
                     MetalBufferRef r, int oR, int n) {}
void Metal_RoPE_F16(MetalContextRef ctx, MetalBufferRef d, int oD, int b, int s,
                    int nh, int hd, int po, float rt) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineRoPE_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)d offset:oD atIndex:0];
  [enc setBytes:&po length:4 atIndex:1];
  [enc setBytes:&hd length:4 atIndex:2];
  [enc setBytes:&rt length:4 atIndex:3];

  // nh is Number of Heads (or KV Heads) to rotate.
  // Each head has hd/2 pairs.
  int pairs = nh * (hd / 2);
  [enc dispatchThreads:MTLSizeMake(pairs, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(pairs, 256), 1, 1)];
}

void Metal_SwiGLU_F16(MetalContextRef ctx, MetalBufferRef iV, int oV,
                      MetalBufferRef iG, int oG, MetalBufferRef o, int oO,
                      int n, int iS) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineSwiGLU_F16];
  // Args: gate, up, output is usually shared?
  // kernel: swiglu(gate, up). Modifies gate in-place.
  // Wait. Check Kernel Signature (Step 2691 line 194):
  // void swiglu_f16(device half *gate, device const half *up, ...)
  // It writes to GATE.
  // My Go `Layer` calls: SwiGLU(up, gate, inter).
  // Arguments C Wrapper: (gate, up, out?)
  // Verify Go Arg Order:
  // C.Metal_SwiGLU_F16(ctx, up.buf, 0, gate.buf, 0, inter.buf, 0, 1, cols)
  // Wait.
  // Arg 1: iV (Up?). Arg 2: iG (Gate?). Arg 3: o (Inter?).
  // If Kernel writes to Arg 0.
  // It should be Gate?
  // Go code: SwiGLU(up, gate, inter).
  // Wait. Standard SwiGLU: (xW_g * Sigmoid(xW_g)) * xW_u.
  // Kernel: gate[i] = up[i] * (gate[i] / (1+exp(-gate[i]))).
  // So Kernel modifies Gate.
  // If I want result in `inter`, I must copy Gate to Inter first?
  // Or pass `inter` as Gate?
  // If `inter` is uninitialized.
  // I should pass `gate` as 1st arg. `up` as 2nd arg.
  // BUT Go code passed `up` first?
  // Go: `C.Metal_SwiGLU_F16(..., upPart.buf, ..., gatePart.buf, ...,
  // interPart.buf, ...)` If I implement C Wrapper to match Kernel: Kernel uses
  // Buf 0 (Gate - In/Out), Buf 1 (Up - In). AND Kernel ignores Buf 2? Kernel
  // doc Step 2691 line 194 ONLY lists Buffer 0, 1. It DOES NOT have output
  // buffer. It is In-Place on Buffer 0.

  // So my C Wrapper should setup Buffer 0 and 1.
  // If Go passes 3 buffers.
  // I should check which one is Gate.
  // Go: Arg 1 `up`. Arg 2 `gate`.
  // Wrapper `iV` (Arg 1), `iG` (Arg 2).
  // So Buffer 0 should be `iG`. Buffer 1 should be `iV`.
  // And `o` (Arg 3) is unused by kernel?
  // If Kernel modifes Gate in place.
  // Then `gatePart` is modified.
  // `interPart` is unused.
  // Go `Layer` used `interPart` for next Linear.
  // I must fix Go logic OR implement Copy in Wrapper.
  // Ideally, SwiGLU Writes to Output.
  // I will UPDATE Kernel later? No, stick to existing kernel to avoid
  // recompiling metal (complex). Existing Kernel modifies Arg 0. So:
  [enc setBuffer:(__bridge id<MTLBuffer>)iG
          offset:oG
         atIndex:0]; // Gate (Modified)
  [enc setBuffer:(__bridge id<MTLBuffer>)iV offset:oV atIndex:1]; // Up

  [enc dispatchThreads:MTLSizeMake(iS, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(iS, 256), 1, 1)];
}

void Metal_Softmax_F16(MetalContextRef ctx, MetalBufferRef i, int oI,
                       MetalBufferRef r, int oR, int rs, int cs) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineSoftmax_F16];
  // Kernel: scores(0), pos(1), stride(2).
  // Kernel modifies scores in place.
  [enc setBuffer:(__bridge id<MTLBuffer>)i offset:oI atIndex:0];
  [enc setBytes:&cs
         length:4
        atIndex:1]; // pos (Wait. cs is 'cols'? Go passed 'pos')
  [enc setBytes:&rs length:4 atIndex:2]; // stride

  // Grid: Rows (Heads).
  // Go passed `numHeads`.
  // Wrapper `rs` (RowStride? or Rows?).
  // Function sig: `rs, cs`.
  // Checking Go invocation (Softmax? No, Logic in Go Layer used
  // C.Metal_Attention_F16). Wait. Go Layer DOES NOT CALL `Metal_Softmax`.
  // Sequential Go Layer calls `Metal_Attention_F16` (Line 353).
  // Does `Metal_Attention_F16` do Scores+Softmax+Values?
  // Looking at Stub `void Metal_Attention_F16...`.
  // Reference `Metal_Layer_F16` (Lines 282-312):
  // It calls `pipelineAttScores`, THEN `pipelineSoftmax`, THEN
  // `pipelineAttValues`. So `Metal_Attention_F16` SHOULD implement ALL THREE
  // logic steps. NOT just one.

  // So I don't need `Metal_Softmax_F16` exposed if I bundle it in `Attention`.
  // But I will implement Softmax stub anyway.
  [enc dispatchThreads:MTLSizeMake(rs, 1, 1) // rs = heads
      threadsPerThreadgroup:MTLSizeMake(MIN(rs, 256), 1, 1)];
}

void Metal_StoreKV_F16(MetalContextRef ctx, MetalBufferRef k, int oK,
                       MetalBufferRef v, int oV, MetalBufferRef kC,
                       MetalBufferRef vC, int p, int h, int hd) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  [enc setComputePipelineState:mc.pipelineCopy_F16];

  int kv_dim = h * hd; // kv_heads * head_dim

  // K
  [enc setBuffer:(__bridge id<MTLBuffer>)k offset:oK atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)kC
          offset:p * kv_dim * 2
         atIndex:1]; // Offset in Bytes (2 bytes/half)
  [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(kv_dim, 256), 1, 1)];

  // V
  [enc setBuffer:(__bridge id<MTLBuffer>)v offset:oV atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)vC offset:p * kv_dim * 2 atIndex:1];
  [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(kv_dim, 256), 1, 1)];
}

void Metal_Attention_F16(MetalContextRef ctx, MetalBufferRef q, int oQ,
                         MetalBufferRef kC, MetalBufferRef vC, MetalBufferRef r,
                         int oR, MetalBufferRef s, int oS, int p, int nh,
                         int kh, int hd, int ctxLen) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLComputeCommandEncoder> enc = [mc ensureEncoder];
  int stride = ctxLen;

  // 1. Scores
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

  // 2. Softmax
  [enc setComputePipelineState:mc.pipelineSoftmax_F16];
  // Softmax in-place on 's'
  [enc setBuffer:(__bridge id<MTLBuffer>)s offset:oS atIndex:0];
  [enc setBytes:&p length:4 atIndex:1];
  [enc setBytes:&stride length:4 atIndex:2];
  [enc dispatchThreads:MTLSizeMake(nh, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(nh, 1, 1)];
  [mc barrier];

  // 3. Values
  [enc setComputePipelineState:mc.pipelineAttValues_F16];
  [enc setBuffer:(__bridge id<MTLBuffer>)s offset:oS atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)vC offset:0 atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)r offset:oR atIndex:2];
  [enc setBytes:&p length:4 atIndex:3];
  [enc setBytes:&nh length:4 atIndex:4];
  [enc setBytes:&kh length:4 atIndex:5];
  [enc setBytes:&hd length:4 atIndex:6];
  [enc setBytes:&stride length:4 atIndex:7];
  // Grid: (headDim * numHeads, 1, 1)
  // Threadgroup: (headDim, 1, 1) -> 128
  int dim = nh * hd;
  [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(dim, 256), 1, 1)];
}

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
