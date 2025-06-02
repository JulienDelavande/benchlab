// test_cuda.cu
#include <cuda_runtime.h>
#include <iostream>

__global__ void square(float* x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    x[idx] *= x[idx];
}

int main() {
    float* d;
    cudaMalloc(&d, sizeof(float) * 1024);
    square<<<1, 1024>>>(d);
    cudaDeviceSynchronize();
    cudaFree(d);
    std::cout << "Done" << std::endl;
    return 0;
}
