#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_DIM 256
#define MASK 0xffffffff

__global__ void compute_loss(const float* logits, const int* true_labels, float* loss, int N, int C) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= N) return;

    float local_sum = 0.0f;

    for (int i = tid; i < C; i += blockDim.x) {
        local_sum += __expf(logits[row * C + i]);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    __shared__ float warp_sums[BLOCK_DIM / 32];
    if (tid % 32 == 0) {
        warp_sums[tid / 32] = local_sum;
    }
    __syncthreads();

    float sum_exp = 0.0f;
    if (tid < BLOCK_DIM / 32) {
        sum_exp = warp_sums[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
        }
    }

    if (tid == 0) {
        float true_logit = logits[row * C + true_labels[row]];
        float sample_loss = -true_logit + logf(sum_exp);
        atomicAdd(loss, sample_loss);
    }
}
__global__ void average_loss(float* loss, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *loss /= N;
    }
}
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    cudaMemset(loss, 0, sizeof(float));

    int blocks = N;
    int threads = 256;
    compute_loss<<<blocks, threads>>>(logits, true_labels, loss, N, C);
    cudaDeviceSynchronize();  
    average_loss<<<1, 1>>>(loss, N); 
}

