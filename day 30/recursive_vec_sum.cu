#include <iostream>
#include <cuda_runtime.h>

#define N 1024
#define TPB 256  

__global__ void recursiveSum(int *input, int *output, int size) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < size) ? input[i] : 0;
    __syncthreads();


    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }


    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }


    if (blockIdx.x == 0 && gridDim.x > 1 && tid == 0) {
        int newSize = gridDim.x;
        int newBlocks = (newSize + TPB - 1) / TPB;
        recursiveSum<<<newBlocks, TPB, TPB * sizeof(int)>>>(output, output, newSize);
    }
}

int main() {
    int h_input[N];
    for (int i = 0; i < N; ++i) h_input[i] = 1; 
    int *d_input;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (N + TPB - 1) / TPB;

    recursiveSum<<<blocks, TPB, TPB * sizeof(int)>>>(d_input, d_input, N);
    cudaDeviceSynchronize();

    int result = 0;
    cudaMemcpy(&result, d_input, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Total sum: " << result << std::endl;

    cudaFree(d_input);
    return 0;
}
