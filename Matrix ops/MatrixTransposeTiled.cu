#include "solve.h"
#include <cuda_runtime.h>

#define TILE_WIDTH 8

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH + 1]; 
    int row_in = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col_in = blockIdx.x * TILE_WIDTH + threadIdx.x;

    if (row_in < rows && col_in < cols) {
        tile[threadIdx.y][threadIdx.x] = input[row_in * cols + col_in];
    }

    __syncthreads();

    int row_out = blockIdx.x * TILE_WIDTH + threadIdx.y;
    int col_out = blockIdx.y * TILE_WIDTH + threadIdx.x;

    if (row_out < cols && col_out < rows) {
        output[row_out * rows + col_out] = tile[threadIdx.x][threadIdx.y];
    }
}
void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((cols + TILE_WIDTH - 1) / TILE_WIDTH,
                       (rows + TILE_WIDTH - 1) / TILE_WIDTH);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

