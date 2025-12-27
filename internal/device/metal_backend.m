// go:build metal
//  +build metal

#import "metal_bridge.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface MetalWrapper : NSObject
@property(strong) id<MTLDevice> device;
@property(strong) id<MTLCommandQueue> commandQueue;
@property(strong) id<MTLLibrary> library;

// Pipelines
@property(strong) id<MTLComputePipelineState> pipelineAdd_F16;
@property(strong) id<MTLComputePipelineState> pipelineScale_F16;
@property(strong) id<MTLComputePipelineState> pipelineRMSNorm_F16;
@property(strong) id<MTLComputePipelineState> pipelineRoPE_F16;
@property(strong) id<MTLComputePipelineState> pipelineSwiGLU_F16;
@property(strong) id<MTLComputePipelineState> pipelineSoftmax_F16;
@property(strong) id<MTLComputePipelineState> pipelineEmbedding_F16;
@property(strong) id<MTLComputePipelineState> pipelineStoreKV_F16;
@property(strong) id<MTLComputePipelineState> pipelineAttention_F16;
@property(strong) id<MTLComputePipelineState> pipelineRMSNormLinear_F16;

@property(strong) id<MTLCommandBuffer> currentCommandBuffer;
@property(strong) id<MTLComputeCommandEncoder> currentEncoder;
@end

@implementation MetalWrapper
- (void)stopEncoder {
  if (self.currentEncoder) {
    [self.currentEncoder endEncoding];
    self.currentEncoder = nil;
  }
}

- (void)ensureEncoder {
  if (!self.currentCommandBuffer) {
    self.currentCommandBuffer = [self.commandQueue commandBuffer];
  }
  if (!self.currentEncoder) {
    self.currentEncoder = [self.currentCommandBuffer computeCommandEncoder];
  }
}

- (void)ensureCommandBuffer {
  if (!self.currentCommandBuffer) {
    self.currentCommandBuffer = [self.commandQueue commandBuffer];
  }
}

- (void)flush {
  [self stopEncoder];
  if (self.currentCommandBuffer) {
    [self.currentCommandBuffer commit];
    // Remove synchronous wait - let GPU work asynchronously
    self.currentCommandBuffer = nil;
  }
}
@end

static id<MTLComputePipelineState> loadPipeline(MetalWrapper *ctx,
                                                NSString *name) {
  id<MTLFunction> fn = [ctx.library newFunctionWithName:name];
  if (!fn) {
    NSLog(@"Failed to find function: %@", name);
    return nil;
  }
  return [ctx.device newComputePipelineStateWithFunction:fn error:nil];
}

MetalContextRef Metal_Init(const char *libSource) {
  MetalWrapper *ctx = [[MetalWrapper alloc] init];
  ctx.device = MTLCreateSystemDefaultDevice();
  if (!ctx.device)
    return NULL;
  ctx.commandQueue = [ctx.device newCommandQueue];

  NSError *error = nil;
  NSString *src = [NSString stringWithUTF8String:libSource];
  ctx.library = [ctx.device newLibraryWithSource:src options:nil error:&error];
  if (error) {
    NSLog(@"Metal compilation error: %@", error);
    return NULL;
  }

  ctx.pipelineAdd_F16 = loadPipeline(ctx, @"add_kernel_f16");
  ctx.pipelineScale_F16 = loadPipeline(ctx, @"scale_kernel_f16");
  ctx.pipelineRMSNorm_F16 = loadPipeline(ctx, @"rmsnorm_kernel_f16");
  ctx.pipelineRoPE_F16 = loadPipeline(ctx, @"rope_kernel_f16");
  ctx.pipelineSwiGLU_F16 = loadPipeline(ctx, @"swiglu_kernel_f16");
  ctx.pipelineSoftmax_F16 = loadPipeline(ctx, @"softmax_kernel_f16");
  ctx.pipelineEmbedding_F16 = loadPipeline(ctx, @"embedding_kernel_f16");
  ctx.pipelineStoreKV_F16 = loadPipeline(ctx, @"kv_store_f16");
  ctx.pipelineAttention_F16 = loadPipeline(ctx, @"attention_f16");
  ctx.pipelineRMSNormLinear_F16 = loadPipeline(ctx, @"rmsnorm_linear_f16");

  return (__bridge_retained MetalContextRef)ctx;
}

void Metal_Free(MetalContextRef ctx) {
  if (ctx) {
    MetalWrapper *wrapper = (__bridge_transfer MetalWrapper *)ctx;
    [wrapper flush];
    wrapper = nil;
  }
}

void Metal_Synchronize(MetalContextRef ctx) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  [c stopEncoder];
  if (c.currentCommandBuffer) {
    [c.currentCommandBuffer commit];
    [c.currentCommandBuffer waitUntilCompleted]; // MUST wait for ToHost
    c.currentCommandBuffer = nil;
  }
}

MetalBufferRef Metal_Alloc(MetalContextRef ctx, int size) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLBuffer> buf =
      [mc.device newBufferWithLength:size options:MTLResourceStorageModeShared];
  return (__bridge_retained MetalBufferRef)buf;
}

void Metal_FreeBuffer(MetalContextRef ctx, MetalBufferRef buf) {
  if (buf) {
    id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)buf;
    buffer = nil;
  }
}

void Metal_CopyToDevice(MetalBufferRef buf, int offset, const void *data,
                        int size) {
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buf;
  memcpy((char *)[buffer contents] + offset, data, size);
}

void Metal_CopyToHost(MetalBufferRef buf, int offset, void *data, int size) {
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buf;
  memcpy(data, (char *)[buffer contents] + offset, size);
}

void *Metal_GetBufferContents(MetalBufferRef buf) {
  return [(__bridge id<MTLBuffer>)buf contents];
}

void Metal_ZeroBuffer(MetalBufferRef buf, int offset, int size) {
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buf;
  memset((char *)[buffer contents] + offset, 0, size);
}

// ================= KERNEL DISPATCH =================

#define ENCODE(wrapper, pipeline)                                              \
  [wrapper ensureEncoder];                                                     \
  [wrapper.currentEncoder setComputePipelineState:wrapper.pipeline];

void Metal_Add_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                   MetalBufferRef b, int offB, MetalBufferRef result,
                   int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineAdd_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)b offset:offB atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];
  [c.currentEncoder dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(count, 512), 1, 1)];
}

void Metal_Scale_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                     uint16_t val, MetalBufferRef result, int offRes,
                     int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineScale_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBytes:&val length:2 atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];
  [c.currentEncoder dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(count, 512), 1, 1)];
}

void Metal_Embedding_F16(MetalContextRef ctx, MetalBufferRef weights, int offW,
                         MetalBufferRef result, int offRes, int rowIdx,
                         int cols) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineEmbedding_F16);

  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)weights
                       offset:offW
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];
  [c.currentEncoder setBytes:&rowIdx length:4 atIndex:2];
  [c.currentEncoder setBytes:&cols length:4 atIndex:3];

  [c.currentEncoder dispatchThreads:MTLSizeMake(cols, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(cols, 512), 1, 1)];
}

void Metal_RMSNorm_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                       MetalBufferRef weight, int offWeight,
                       MetalBufferRef result, int offRes, int rows, int cols,
                       float eps) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineRMSNorm_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)input
                       offset:offIn
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)weight
                       offset:offWeight
                      atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];
  [c.currentEncoder setBytes:&cols length:4 atIndex:3];
  [c.currentEncoder setBytes:&eps length:4 atIndex:4];
  [c.currentEncoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void Metal_RoPE_F16(MetalContextRef ctx, MetalBufferRef data, int offData,
                    int batchSize, int seqLen, int numHeads, int headDim,
                    int posOffset, float ropeTheta) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineRoPE_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)data
                       offset:offData
                      atIndex:0];
  [c.currentEncoder setBytes:&headDim length:4 atIndex:1];
  [c.currentEncoder setBytes:&numHeads length:4 atIndex:2];
  [c.currentEncoder setBytes:&seqLen length:4 atIndex:3];
  [c.currentEncoder setBytes:&posOffset length:4 atIndex:4];
  [c.currentEncoder setBytes:&ropeTheta length:4 atIndex:5];
  [c.currentEncoder dispatchThreads:MTLSizeMake(headDim / 2, numHeads,
                                                batchSize * seqLen)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}

void Metal_SwiGLU_F16(MetalContextRef ctx, MetalBufferRef inputVal, int offVal,
                      MetalBufferRef inputGate, int offGate,
                      MetalBufferRef output, int offOut, int n, int interSize) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineSwiGLU_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)inputVal
                       offset:offVal
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)inputGate
                       offset:offGate
                      atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)output
                       offset:offOut
                      atIndex:2];
  [c.currentEncoder setBytes:&interSize length:4 atIndex:3];
  [c.currentEncoder dispatchThreads:MTLSizeMake(interSize, n, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}

void Metal_Softmax_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                       MetalBufferRef result, int offRes, int rows, int cols) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineSoftmax_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)input
                       offset:offIn
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];
  [c.currentEncoder setBytes:&cols length:4 atIndex:2];
  [c.currentEncoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// ================= MPS MatMul =================

void Metal_MatMul_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                      bool transA, MetalBufferRef b, int offB, bool transB,
                      MetalBufferRef c, int offC, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  [mc stopEncoder];
  [mc ensureCommandBuffer];

  MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transA ? K : M)
                       columns:(transA ? M : K)rowBytes:(transA ? M : K) * 2
                      dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *dB = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transB ? N : K)
                       columns:(transB ? K : N)rowBytes:(transB ? K : N) * 2
                      dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *dC =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                            columns:N
                                           rowBytes:N * 2
                                           dataType:MPSDataTypeFloat16];

  MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a
                                             offset:offA
                                         descriptor:dA];
  MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b
                                             offset:offB
                                         descriptor:dB];
  MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)c
                                             offset:offC
                                         descriptor:dC];

  MPSMatrixMultiplication *mul =
      [[MPSMatrixMultiplication alloc] initWithDevice:mc.device
                                        transposeLeft:transA
                                       transposeRight:transB
                                           resultRows:M
                                        resultColumns:N
                                      interiorColumns:K
                                                alpha:1.0
                                                 beta:0.0];
  [mul encodeToCommandBuffer:mc.currentCommandBuffer
                  leftMatrix:mA
                 rightMatrix:mB
                resultMatrix:mC];
}

void Metal_BatchedMatMul_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                             int strideA, bool transA, MetalBufferRef b,
                             int offB, int strideB, bool transB,
                             MetalBufferRef c, int offC, int strideC, int M,
                             int N, int K, int batchCount) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  [mc stopEncoder];
  [mc ensureCommandBuffer];

  MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transA ? K : M)
                       columns:(transA ? M : K)rowBytes:(transA ? M : K) * 2
                      dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *dB = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transB ? N : K)
                       columns:(transB ? K : N)rowBytes:(transB ? K : N) * 2
                      dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *dC =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                            columns:N
                                           rowBytes:N * 2
                                           dataType:MPSDataTypeFloat16];

  MPSMatrixMultiplication *mul =
      [[MPSMatrixMultiplication alloc] initWithDevice:mc.device
                                        transposeLeft:transA
                                       transposeRight:transB
                                           resultRows:M
                                        resultColumns:N
                                      interiorColumns:K
                                                alpha:1.0
                                                 beta:0.0];

  for (int i = 0; i < batchCount; i++) {
    MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a
                                               offset:offA + i * strideA
                                           descriptor:dA];
    MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b
                                               offset:offB + i * strideB
                                           descriptor:dB];
    MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)c
                                               offset:offC + i * strideC
                                           descriptor:dC];

    [mul encodeToCommandBuffer:mc.currentCommandBuffer
                    leftMatrix:mA
                   rightMatrix:mB
                  resultMatrix:mC];
  }
}
void Metal_StoreKV_F16(MetalContextRef ctx, MetalBufferRef k, int offK,
                       MetalBufferRef v, int offV, MetalBufferRef kCache,
                       MetalBufferRef vCache, int pos, int heads, int headDim) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineStoreKV_F16);

  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)k offset:offK atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)v offset:offV atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)kCache
                       offset:0
                      atIndex:2];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)vCache
                       offset:0
                      atIndex:3];
  [c.currentEncoder setBytes:&pos length:4 atIndex:4];
  [c.currentEncoder setBytes:&heads length:4 atIndex:5];
  [c.currentEncoder setBytes:&headDim length:4 atIndex:6];

  [c.currentEncoder dispatchThreads:MTLSizeMake(heads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(heads, 512), 1, 1)];
}

void Metal_Attention_F16(MetalContextRef ctx, MetalBufferRef q, int offQ,
                         MetalBufferRef kCache, MetalBufferRef vCache,
                         MetalBufferRef result, int offRes, int pos,
                         int numHeads, int kvHeads, int headDim) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineAttention_F16);

  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)q offset:offQ atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)kCache
                       offset:0
                      atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)vCache
                       offset:0
                      atIndex:2];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:3];
  [c.currentEncoder setBytes:&pos length:4 atIndex:4];
  [c.currentEncoder setBytes:&numHeads length:4 atIndex:5];
  [c.currentEncoder setBytes:&kvHeads length:4 atIndex:6];
  [c.currentEncoder setBytes:&headDim length:4 atIndex:7];

  [c.currentEncoder dispatchThreads:MTLSizeMake(numHeads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(numHeads, 512), 1, 1)];
}

void Metal_RMSNormLinear_F16(MetalContextRef ctx, MetalBufferRef input,
                             int offIn, MetalBufferRef normWeight,
                             int offNormWeight, MetalBufferRef weight,
                             int offWeight, MetalBufferRef result, int offRes,
                             int inDim, int outDim, float eps) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineRMSNormLinear_F16);

  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)input
                       offset:offIn
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)normWeight
                       offset:offNormWeight
                      atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)weight
                       offset:offWeight
                      atIndex:2];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:3];
  [c.currentEncoder setBytes:&inDim length:4 atIndex:4];
  [c.currentEncoder setBytes:&outDim length:4 atIndex:5];
  [c.currentEncoder setBytes:&eps length:4 atIndex:6];

  [c.currentEncoder dispatchThreads:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
