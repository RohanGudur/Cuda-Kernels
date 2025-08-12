#include <cuda_runtime.h>

#define BLOCK_DIM 256

__global__ void vector_add(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N){
        C[i] = A[i] + B[i];
    }
    
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = BLOCK_DIM;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}