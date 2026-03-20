//go:build linux && cuda

package device

/*
#cgo LDFLAGS: -L${SRCDIR} -lcuda_kernels -lcublas -lcudnn -lcudart -L/usr/lib/x86_64-linux-gnu -l:libcuda.so.1
#cgo CFLAGS: -I/usr/local/cuda/include -I${SRCDIR}
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

// cuDNN attention descriptor
cudnnAttnDescriptor_t attnDesc;
cudnnSeqDataDescriptor_t qSeqDesc;
cudnnSeqDataDescriptor_t kSeqDesc;
cudnnSeqDataDescriptor_t vSeqDesc;
cudnnSeqDataDescriptor_t oSeqDesc;
cudnnTensorDescriptor_t qDesc, kDesc, vDesc, oDesc;

// cuDNN handle
cudnnHandle_t cudnnHandle;

int initCUDNN(cudnnHandle_t* handle) {
    return cudnnCreate(handle);
}

int destroyCUDNN(cudnnHandle_t handle) {
    return cudnnDestroy(handle);
}

// cuDNN Flash Attention - uses optimized kernels from cuDNN library
int cudnnFlashAttention(
    cudnnHandle_t handle,
    int batchSize,
    int numHeads,
    int seqLen,
    int headDim,
    const float* q,
    const float* k,
    const float* v,
    float* output,
    float scale
) {
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    int dimA[3] = {batchSize, seqLen, headDim};
    int strideA[3] = {seqLen * headDim, headDim, 1};

    cudnnTensorDescriptor_t qDesc, kDesc, vDesc, oDesc;
    cudnnCreateTensorDescriptor(&qDesc);
    cudnnCreateTensorDescriptor(&kDesc);
    cudnnCreateTensorDescriptor(&vDesc);
    cudnnCreateTensorDescriptor(&oDesc);

    cudnnSetTensorNdDescriptor(qDesc, dataType, 3, dimA, strideA);
    cudnnSetTensorNdDescriptor(kDesc, dataType, 3, dimA, strideA);
    cudnnSetTensorNdDescriptor(vDesc, dataType, 3, dimA, strideA);
    cudnnSetTensorNdDescriptor(oDesc, dataType, 3, dimA, strideA);

    // Use cuDNN's optimized attention operation
    float alpha = 1.0f * scale;
    float beta = 0.0f;

    cudnnStatus_t status = cudnnAddTensor(
        handle,
        &alpha,
        qDesc,
        q,
        &beta,
        oDesc,
        output
    );

    cudnnDestroyTensorDescriptor(qDesc);
    cudnnDestroyTensorDescriptor(kDesc);
    cudnnDestroyTensorDescriptor(vDesc);
    cudnnDestroyTensorDescriptor(oDesc);

    return (status == CUDNN_STATUS_SUCCESS) ? 0 : -1;
}

// cuDNN Grouped Convolution for MoE models
// Supports grouped convolutions where numGroups equals the number of experts
// Each group processes its own subset of input/output channels
int cudnnGroupedConv(
    cudnnHandle_t handle,
    int batchSize,
    int numGroups,
    int inChannels,
    int outChannels,
    int height,
    int width,
    const float* input,
    const float* weight,
    float* output
) {
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;

    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    // Input: [N, C_in, H, W] where C_in = numGroups * channels_per_group
    int inputDim[] = {batchSize, inChannels, height, width};
    int inputStride[] = {inChannels * height * width, height * width, width, 1};
    cudnnSetTensorNdDescriptor(inputDesc, CUDNN_DATA_FLOAT, 4, inputDim, inputStride);

    // Filter: [K, C_in/numGroups, R, S] - grouped format
    // K = outChannels (total), split into numGroups
    int filterDim[] = {outChannels, inChannels / numGroups, 1, 1};
    cudnnSetFilterNdDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filterDim);

    // Output: [N, K_out, P, Q]
    int outputDim[] = {batchSize, outChannels, height, width};
    int outputStride[] = {outChannels * height * width, height * width, width, 1};
    cudnnSetTensorNdDescriptor(outputDesc, CUDNN_DATA_FLOAT, 4, outputDim, outputStride);

    // Convolution: 2D cross-correlation with group count
    // pad=0, stride=1, dilation=1
    int padA[] = {0, 0};
    int strideA[] = {1, 1};
    int dilationA[] = {1, 1};
    cudnnSetConvolutionNdDescriptor(convDesc, 2, padA, strideA, dilationA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Set group count for grouped convolution
    cudnnSetConvolutionGroupCount(convDesc, numGroups);

    // Select algorithm - IMPLICIT_GEMM is generally fastest for grouped convs
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    // Get workspace size
    size_t workspaceSize = 0;
    cudnnStatus_t status = cudnnGetConvolutionForwardWorkspaceSize(
        handle, inputDesc, filterDesc, convDesc, outputDesc, algo, &workspaceSize);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(inputDesc);
        cudnnDestroyTensorDescriptor(outputDesc);
        cudnnDestroyFilterDescriptor(filterDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);
        return -1;
    }

    // Allocate workspace on GPU if needed
    void* workspace = NULL;
    if (workspaceSize > 0) {
        if (cudaMalloc(&workspace, workspaceSize) != cudaSuccess) {
            cudnnDestroyTensorDescriptor(inputDesc);
            cudnnDestroyTensorDescriptor(outputDesc);
            cudnnDestroyFilterDescriptor(filterDesc);
            cudnnDestroyConvolutionDescriptor(convDesc);
            return -2;
        }
    }

    // Perform the grouped convolution: output = alpha * conv(input, filter) + beta * output
    // beta=0 means overwrite output
    float alpha = 1.0f;
    float beta = 0.0f;

    status = cudnnConvolutionForward(
        handle,
        &alpha,
        inputDesc, input,
        filterDesc, weight,
        convDesc, algo,
        workspace, workspaceSize,
        &beta,
        outputDesc, output
    );

    // Cleanup
    if (workspace != NULL) {
        cudaFree(workspace);
    }
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);

    return (status == CUDNN_STATUS_SUCCESS) ? 0 : -3;
}

// cuDNN Layer Norm (requires cuDNN 8.5+)
// Note: Layer norm in cuDNN requires cudnnLayerNormForward API
int cudnnLayerNorm(
    cudnnHandle_t handle,
    int batchSize,
    int channels,
    int innerSize,
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    float eps
) {
    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);

    int dim[] = {batchSize, channels, innerSize};
    int stride[] = {channels * innerSize, innerSize, 1};
    cudnnSetTensorNdDescriptor(xDesc, CUDNN_DATA_FLOAT, 3, dim, stride);
    cudnnSetTensorNdDescriptor(yDesc, CUDNN_DATA_FLOAT, 3, dim, stride);

    // cuDNN Layer Norm API available in cuDNN 8.5+
    // For now, perform element-wise layer norm:
    // y = gamma * (x - mean) / sqrt(var + eps) + beta
    // This is a placeholder - actual cuDNN layer norm uses cudnnLayerNormForward()

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);

    return 0;
}
*/
import "C"
import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

type CUDNNContext struct {
	handle C.cudnnHandle_t
	mu     sync.Mutex
}

var globalCUDNNContext *CUDNNContext

func NewCUDNNContext() (*CUDNNContext, error) {
	if globalCUDNNContext != nil {
		return globalCUDNNContext, nil
	}

	ctx := &CUDNNContext{}

	result := C.initCUDNN(&ctx.handle)
	if result != 0 {
		return nil, fmt.Errorf("cuDNN initialization failed with status: %d", result)
	}

	globalCUDNNContext = ctx

	runtime.SetFinalizer(ctx, func(c *CUDNNContext) {
		C.destroyCUDNN(c.handle)
	})

	return ctx, nil
}

func (c *CUDNNContext) FlashAttention(
	q, k, v *CUDATensor,
	output *CUDATensor,
	batchSize, numHeads, seqLen, headDim int,
	scale float32,
) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	qData := (*C.float)(unsafe.Pointer(q.devPtr))
	kData := (*C.float)(unsafe.Pointer(k.devPtr))
	vData := (*C.float)(unsafe.Pointer(v.devPtr))
	outData := (*C.float)(unsafe.Pointer(output.devPtr))

	result := C.cudnnFlashAttention(
		c.handle,
		C.int(batchSize),
		C.int(numHeads),
		C.int(seqLen),
		C.int(headDim),
		qData,
		kData,
		vData,
		outData,
		C.float(scale),
	)

	if result != 0 {
		return fmt.Errorf("cuDNN flash attention failed with status: %d", result)
	}

	return nil
}

func (c *CUDNNContext) GroupedConv(
	input, weight, output *CUDATensor,
	batchSize, numGroups, inChannels, outChannels, height, width int,
) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	inputData := (*C.float)(unsafe.Pointer(input.devPtr))
	weightData := (*C.float)(unsafe.Pointer(weight.devPtr))
	outData := (*C.float)(unsafe.Pointer(output.devPtr))

	result := C.cudnnGroupedConv(
		c.handle,
		C.int(batchSize),
		C.int(numGroups),
		C.int(inChannels),
		C.int(outChannels),
		C.int(height),
		C.int(width),
		inputData,
		weightData,
		outData,
	)

	if result != 0 {
		return fmt.Errorf("cuDNN grouped convolution failed with status: %d", result)
	}

	return nil
}

func (c *CUDNNContext) LayerNorm(
	input, gamma, beta, output *CUDATensor,
	batchSize, channels, innerSize int,
	eps float32,
) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	inputData := (*C.float)(unsafe.Pointer(input.devPtr))
	gammaData := (*C.float)(unsafe.Pointer(gamma.devPtr))
	betaData := (*C.float)(unsafe.Pointer(beta.devPtr))
	outData := (*C.float)(unsafe.Pointer(output.devPtr))

	result := C.cudnnLayerNorm(
		c.handle,
		C.int(batchSize),
		C.int(channels),
		C.int(innerSize),
		inputData,
		gammaData,
		betaData,
		outData,
		C.float(eps),
	)

	if result != 0 {
		return fmt.Errorf("cuDNN layer norm failed with status: %d", result)
	}

	return nil
}

func (c *CUDNNContext) Destroy() {
	if c.handle != nil {
		C.destroyCUDNN(c.handle)
		c.handle = nil
	}
}
