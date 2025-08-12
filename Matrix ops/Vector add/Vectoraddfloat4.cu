#include <cuda_runtime.h>

#define BLOCK_DIM 256
"""
Optimized vector addition kernel using float4 vectorized loads  
__ldg() for read-only cache utilization and tail handling for non-multiple-of-4 sizes.
This approach reduces memory transactions and improves throughput on memory-bound workloads.

"""
__global__ void vector_add(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int N
){    
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t vec_size = N >> 2; // N / 4

    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);      
    float4* C4 = reinterpret_cast<float4*>(C);

    #pragma unroll
    for (size_t i = tid; i < vec_size; i += stride) {
        float4 va = __ldg(&A4[i]);
        float4 vb = __ldg(&B4[i]);
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;

        C4[i] = vc;
    }

    int tail = N % 4;
    int base = N - tail;

    if (tid < tail) {
        C[base + tid] = __ldg(&A[base + tid]) + __ldg(&B[base + tid]);
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = BLOCK_DIM;
    int blocksPerGrid = max(1, (N / 4 + threadsPerBlock - 1) / threadsPerBlock);
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
