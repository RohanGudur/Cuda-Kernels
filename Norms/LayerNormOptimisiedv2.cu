"""
Implement Layer Normalization over the last 3 dimensions (F, D1, D2) of a 4D tensor.

where the mean E[x]E[x] and variance Var[x]Var[x] are computed over the normalization dimensions (F, D1, D2) for each element in the first dimension (B). γγ and ββ are learnable affine parameters (elementwise scale and shift), and ϵϵ is a small value added to the variance for numerical stability.
Input:

    Tensor XX of shape (B,F,D1,D2)(B,F,D1,D2) (input data)
    Vector gammagamma of shape (F,D1,D2)(F,D1,D2) (scale parameters)
    Vector betabeta of shape (F,D1,D2)(F,D1,D2) (shift parameters)
    Epsilon ϵ (a small float, typically 1e-5)

Output:

    Tensor YY of shape (B,F,D1,D2)(B,F,D1,D2)
"""

#include <cuda_runtime.h>
#include <cmath>

__global__ void Layer_norm(const float* X, const float* gamma, const float* beta, float* Y,size_t B, size_t F, size_t D1, size_t D2) {
    extern __shared__ float shared_sums[];
    int tid = threadIdx.x;
    int batch = blockIdx.x;
    
    size_t total = F * D1 * D2;

    float local_sum = 0.0f;
    for (unsigned i = tid * 4; i < total; i += blockDim.x * 4) {
        unsigned idx = batch * total + i;
        float4 v = reinterpret_cast<const float4*>(X)[idx / 4];
        local_sum += v.x + v.y + v.z + v.w;
    }

    shared_sums[tid] = local_sum;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }
        __syncthreads();
    }

    float mean = shared_sums[0] / total;
    __syncthreads();

    float local_var_sum = 0.0f;
    for (unsigned i = tid * 4; i < total; i += blockDim.x * 4) {
        unsigned idx = batch * total + i;
        float4 v = reinterpret_cast<const float4*>(X)[idx / 4];
        v.x -= mean; v.y -= mean; v.z -= mean; v.w -= mean;
        local_var_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    shared_sums[tid] = local_var_sum;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }
        __syncthreads();
    }

    float var = shared_sums[0] / total;
    float inv_std = rsqrtf(var + 1e-5f);
    __syncthreads();

    for (unsigned i = tid * 4; i < total; i += blockDim.x * 4) {
        unsigned idx = batch * total + i;

        float4 x = reinterpret_cast<const float4*>(X)[idx / 4];
        float4 g = reinterpret_cast<const float4*>(gamma)[i / 4];
        float4 b = reinterpret_cast<const float4*>(beta)[i / 4];

        x.x = (x.x - mean) * inv_std;
        x.y = (x.y - mean) * inv_std;
        x.z = (x.z - mean) * inv_std;
        x.w = (x.w - mean) * inv_std;

        x.x = x.x * g.x + b.x;
        x.y = x.y * g.y + b.y;
        x.z = x.z * g.z + b.z;
        x.w = x.w * g.w + b.w;

        reinterpret_cast<float4*>(Y)[idx / 4] = x;
    }

}

extern "C" void solution(const float* X, const float* gamma, const float* beta, float* Y, size_t B, size_t F, size_t D1, size_t D2) { 
    Layer_norm <<< B , 1024 , 1024 *sizeof(float) >>>(X,gamma,beta,Y,B,F,D1,D2);   
}