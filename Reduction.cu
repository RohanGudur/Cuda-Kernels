#include <cuda_runtime.h>
"""
Reduction can be written using block shared mem reduction but the below is a warp reduction based kernel and is faster 
we can also use a 2 pass kernel instead of atomic add or use cooperative groups 
2 pass kenel is available in the monte carlo kernel 
"""

# define BLOCK_DIM 256

__global__ void reduce(const float* input, float* output, int N) {
    size_t tid = threadIdx.x;
    size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    if (N < 4) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            float sum = 0.0f;
            for (int i = 0; i < N; ++i) {
                sum += input[i];
            }
            *output = sum;
        }
        return;
    }

    size_t vec = N / 4;
    const float4* input4 = reinterpret_cast<const float4*>(input);

    for (size_t i = global_tid; i < vec; i += stride) {
        float4 v = input4[i];
        local_sum += v.x + v.y + v.z + v.w;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    __shared__ float warp_sums[BLOCK_DIM / 32];
    if (tid % 32 == 0) {
        warp_sums[tid / 32] = local_sum;
    }

    __syncthreads();
    if (tid < BLOCK_DIM / 32) {
        local_sum = warp_sums[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
    }

    if (tid == 0) {
        if (blockIdx.x == 0) {
            int tail = N % 4;
            int base = N - tail;
            for (int i = 0; i < tail; ++i) {
                local_sum += input[base + i];
            }
        }

        atomicAdd(output, local_sum);
    }
}


extern "C" void solve(const float* input, float* output, int N) {
    cudaMemset(output, 0, sizeof(float));

    size_t num_blocks = std::max(1, (N / 4 + BLOCK_DIM - 1) / BLOCK_DIM);
    reduce<<<num_blocks, BLOCK_DIM>>>(input, output, N);
}
