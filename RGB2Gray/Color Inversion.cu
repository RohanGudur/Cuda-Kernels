#include <cuda_runtime.h>
"""
Color Inversion
The image is represented as a 1D array of RGBA (Red, Green, Blue, Alpha) values, 
where each component is an 8-bit unsigned integer (unsigned char).
Color inversion is performed by subtracting each color component (R, G, B) from 255. \
The Alpha component should remain unchanged.
The input array image will contain width * height * 4 elements.
The first 4 elements represent the RGBA values of the top-left pixel, 
the next 4 elements represent the pixel to its right, and so on. 
"""
__global__ void invert_kernel(unsigned char* image, int width, int height) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    uchar4* image4 = reinterpret_cast<uchar4*>(image);
    size_t total = (width * height); 

    for (size_t i = tid; i < total; i += stride) {
        uchar4 v = image4[i];

        v.x = 255 - v.x;
        v.y = 255 - v.y;
        v.z = 255 - v.z;

        image4[i] = v;
    }
}


extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}