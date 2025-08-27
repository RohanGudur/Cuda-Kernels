#include <cuda_runtime.h>
#include <float.h>
#define BLOCK_SIZE 256 
#define COARSENING 8
#define MASK 0xffffffff

__global__ void softmax(const float* input, float* block_max_out,
                                                 float* block_sum_out, int N) {
    int block = blockIdx.x * blockDim.x * COARSENING * 2;
    int tid = threadIdx.x;
    int i = block + tid;

    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    // Coarsened tile loop
    for (int tile = 0; tile < COARSENING * 2; tile++) {
        int idx = i + tile * blockDim.x;
        if (idx < N) {
            float x = input[idx];
            if (x > local_max) {
                local_sum = local_sum * __expf(local_max - x) + 1.0f;
                local_max = x;
            } else {
                local_sum += __expf(x - local_max);
            }
        }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float offset_max = __shfl_down_sync(MASK, local_max, offset);
        float offset_sum = __shfl_down_sync(MASK, local_sum, offset);
        if (offset_max > local_max) {
            local_sum = local_sum * __expf(local_max - offset_max) + offset_sum;
            local_max = offset_max;
        } else {
            local_sum = local_sum + offset_sum * __expf(offset_max - local_max);
        }
    }

    // Shared memory for warp leaders
    __shared__ float max_mem[BLOCK_SIZE / 32];
    __shared__ float sum_mem[BLOCK_SIZE / 32];

    if (tid % 32 == 0) {
        max_mem[tid / 32] = local_max;
        sum_mem[tid / 32] = local_sum;
    }

    __syncthreads();

    // Final warp-level reduction within block
    if (tid < BLOCK_SIZE / 32) {
        local_max = max_mem[tid];
        local_sum = sum_mem[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            float offset_max = __shfl_down_sync(MASK, local_max, offset);
            float offset_sum = __shfl_down_sync(MASK, local_sum, offset);
            if (offset_max > local_max) {
                local_sum = local_sum * __expf(local_max - offset_max) + offset_sum;
                local_max = offset_max;
            } else {
                local_sum = local_sum + offset_sum * __expf(offset_max - local_max);
            }
        }
    }
    // per block global write 
    if (tid == 0) {
        block_max_out[blockIdx.x] = local_max;
        block_sum_out[blockIdx.x] = local_sum;
    }
}

__global__ void reduce(const float* d_block_max, const float* d_block_sum,
                                    float* d_global_max_out, float* d_global_sum_out,
                                    int num_blocks) {

    int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    // strided loop
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float x_max = d_block_max[i];
        float x_sum = d_block_sum[i];
        if (x_max > local_max) {
            local_sum = local_sum * __expf(local_max - x_max) + x_sum;
            local_max = x_max;
        } else {
            local_sum = local_sum + x_sum * __expf(x_max - local_max);
        }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float offset_max = __shfl_down_sync(MASK, local_max, offset);
        float offset_sum = __shfl_down_sync(MASK, local_sum, offset);
        if (offset_max > local_max) {
            local_sum = local_sum * __expf(local_max - offset_max) + offset_sum;
            local_max = offset_max;
        } else {
            local_sum = local_sum + offset_sum * __expf(offset_max - local_max);
        }
    }

    __shared__ float max_mem[BLOCK_SIZE / 32];
    __shared__ float sum_mem[BLOCK_SIZE / 32];

    if (tid % 32 == 0) {
        max_mem[tid / 32] = local_max;
        sum_mem[tid / 32] = local_sum;
    }

    __syncthreads();
    // we reduce from 256 to 8 now we jsut use a thread and get it to solve for 8 and updatet global
    if (tid == 0) {
        float final_max = max_mem[0];
        float final_sum = sum_mem[0];
        for (int i = 1; i < BLOCK_SIZE / 32; ++i) {
            float x_max = max_mem[i];
            float x_sum = sum_mem[i];
            if (x_max > final_max) {
                final_sum = final_sum * __expf(final_max - x_max) + x_sum;
                final_max = x_max;
            } else {
                final_sum = final_sum + x_sum * __expf(x_max - final_max);
            }
        }
        *d_global_max_out = final_max;
        *d_global_sum_out = final_sum;
    }
}

__global__ void compute(const float* input, float* output,
                                              const float* d_global_max, const float* d_global_sum, int N) {
    const float global_max = *d_global_max;
    const float global_sum = *d_global_sum;

    int tid = threadIdx.x;
    int segment = blockIdx.x * blockDim.x * COARSENING * 2;
    int i = segment + tid;

    #pragma unroll
    for (int tile = 0; tile < COARSENING * 2; tile++) {
        int idx = i + tile * blockDim.x;
        if (idx < N) {
            output[idx] = __expf(input[idx] - global_max) / global_sum;
        }
    }
}

extern "C" void solve(const float* d_input, float* d_output, int N) {
    int num_blocks = (N + BLOCK_SIZE * COARSENING * 2 - 1) / (BLOCK_SIZE * COARSENING * 2);

    float *d_block_max, *d_block_sum;
    cudaMalloc(&d_block_max, num_blocks * sizeof(float));
    cudaMalloc(&d_block_sum, num_blocks * sizeof(float));

    softmax<<<num_blocks, BLOCK_SIZE>>>(d_input, d_block_max, d_block_sum, N);
 
    float *d_global_max, *d_global_sum;
    cudaMalloc(&d_global_max, sizeof(float));
    cudaMalloc(&d_global_sum, sizeof(float));

    reduce<<<1, BLOCK_SIZE>>>(d_block_max, d_block_sum, d_global_max, d_global_sum, num_blocks);

    compute<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, d_global_max, d_global_sum, N);

}
