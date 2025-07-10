#include "solve.h"
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    const size_t vec_size = N / 4;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    for (size_t i = tid; i < vec_size; i += stride) {
        float4 v = input4[i];

        v.x = fmaxf(0.0f, v.x);
        v.y = fmaxf(0.0f, v.y);
        v.z = fmaxf(0.0f, v.z);
        v.w = fmaxf(0.0f, v.w);

        output4[i] = v;
    }
    if (tid < (N % 4)) {
        int idx = vec_size * 4 + tid;
        output[idx] = fmaxf(0.0f, input[idx]);
    }
    
}

void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N/4 + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
