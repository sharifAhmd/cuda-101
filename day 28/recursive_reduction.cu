#include <iostream>
#define N 1024
#define TPB 256 

/*
Compile with nvcc -rdc=true
*/

__global__ void recursiveReduce(int *input, int *output, int size) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    sdata[tid] = (idx < size) ? input[idx] : 0;
    __syncthreads();


    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }


    if (tid == 0) output[blockIdx.x] = sdata[0];

    if (blockIdx.x == 0 && gridDim.x > 1 && tid == 0) {
        int newBlocks = (gridDim.x + TPB - 1) / TPB;
        int newSize = gridDim.x;
        recursiveReduce<<<newBlocks, TPB, TPB * sizeof(int)>>>(output, output, newSize);
    }
}

int main() {
    int h_input[N];
    for (int i = 0; i < N; i++) h_input[i] = 1; 

    int *d_input;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (N + TPB - 1) / TPB;

    recursiveReduce<<<blocks, TPB, TPB * sizeof(int)>>>(d_input, d_input, N);

    int h_result;
    cudaMemcpy(&h_result, d_input, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sum using dynamic parallelism: " << h_result << "\n";

    cudaFree(d_input);
    return 0;
}
