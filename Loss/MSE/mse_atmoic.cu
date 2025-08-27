#include <cuda_runtime.h>
#include <algorithm>

#define BLOCK_DIM 256

__global__ void reduce(const float* predictions, const float* targets, float* mse, int N) {
    size_t tid = threadIdx.x;
    size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    if (N < 4) {
        if (tid < N) {
            float sum = 0.0f;
            for (int i = 0; i < N; ++i) {
                float diff = predictions[i] - targets[i];
                sum += diff * diff;
            }
            if (tid == 0)
                *mse = sum;
        }
        return;
    }

    size_t vec = N / 4;
    const float4* p4 = reinterpret_cast<const float4*>(predictions);
    const float4* t4 = reinterpret_cast<const float4*>(targets);

    for (size_t i = global_tid; i < vec; i += stride) {
        float4 p = p4[i];
        float4 t = t4[i];

        float dx = p.x - t.x;
        float dy = p.y - t.y;
        float dz = p.z - t.z;
        float dw = p.w - t.w;

        local_sum += dx * dx + dy * dy + dz * dz + dw * dw;
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
                float diff = predictions[base + i] - targets[base + i];
                local_sum += diff * diff;
            }
        }

        atomicAdd(mse, local_sum);
    }
}

extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    cudaMemset(mse, 0, sizeof(float));

    size_t num_blocks = std::max(1, (N / 4 + BLOCK_DIM - 1) / BLOCK_DIM);
    reduce<<<num_blocks, BLOCK_DIM>>>(predictions, targets, mse, N);
    float mse_host;
    cudaMemcpy(&mse_host, mse, sizeof(float), cudaMemcpyDeviceToHost);
    mse_host /= N;
    cudaMemcpy(mse, &mse_host, sizeof(float), cudaMemcpyHostToDevice);
}
