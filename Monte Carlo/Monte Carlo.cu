#include <cuda_runtime.h>

"The below is a two pass reduction based kernel "

#define BLOCK_DIM 256 
#define mask 0xffffffff

__device__ float block_sums[1024]; 

__global__ void monte(const float* y_samples, int n_samples) {
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t vec_size = n_samples / 4;
    float local_sum = 0.0f;

    const float4* Y4 = reinterpret_cast<const float4*>(y_samples); 
    for (int i = tid; i < vec_size; i += stride) {
        float4 v = Y4[i];
        local_sum += v.x + v.y + v.z + v.w;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    __shared__ float warp_sums[BLOCK_DIM / 32]; 

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads(); 

    float block_sum = 0.0f;
    if (warp_id == 0) {
        local_sum = (threadIdx.x < BLOCK_DIM / 32) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(mask, local_sum, offset);
        }
        if (lane_id == 0) {
            block_sum = local_sum;
            if (blockIdx.x == 0) {
                int tail = n_samples % 4; 
                for (int i = n_samples - tail; i < n_samples; ++i) {
                    block_sum += y_samples[i];
                }
            }

            block_sums[blockIdx.x] = block_sum;
        }
    }
}

__global__ void reduce_partial(float* result, float a, float b, int n_samples, int num_blocks) {
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    float val = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        val += block_sums[i];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    __shared__ float warp_sums[BLOCK_DIM / 32];
    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }

    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0) {
        val = (tid < BLOCK_DIM / 32) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }

        if (lane_id == 0) {
            *result = (b - a) * (val / n_samples);
        }
    }
}

extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    int threadsPerBlock = BLOCK_DIM;
    int blocksPerGrid = (n_samples / 4 + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemset(result, 0, sizeof(float));

    monte<<<blocksPerGrid, threadsPerBlock>>>(y_samples, n_samples); 
    cudaDeviceSynchronize();

    reduce_partial<<<1, threadsPerBlock>>>(result, a , b , n_samples , blocksPerGrid);
}
