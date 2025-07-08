#include <iostream>
#include <cuda_runtime.h>


__global__ void dkernel(){

    printf("HelloWorld\n");
}

int main(){
    dkernel <<<1 , 1>>>();
     cudaDeviceSynchronize();
    return 0;
}

