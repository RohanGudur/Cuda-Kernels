#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C,
                                           int M, int N, int K) {
    __shared__ float subTileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float subTileB[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * BLOCK_SIZE + ty;
    int Col = bx * BLOCK_SIZE + tx;

    float PValue = 0.0f;

    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int m = 0; m < numTiles; m++) {
        int tiledColA = m * BLOCK_SIZE + tx;
        int tiledRowB = m * BLOCK_SIZE + ty;

        subTileA[ty][tx] = (Row < M && tiledColA < N) ? A[Row * N + tiledColA] : 0.0f;
        subTileB[ty][tx] = (tiledRowB < N && Col < K) ? B[tiledRowB * K + Col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            PValue += subTileA[ty][k] * subTileB[k][tx];
        }

        __syncthreads();
    }

    if (Row < M && Col < K) {
        C[Row * K + Col] = PValue;
    }
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
