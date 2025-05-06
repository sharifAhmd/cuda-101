#include <iostream>

#define N 16
#define RADIUS 1  

/**
 * Performs a 1D convolution operation on the input array.
 * Use shared memory for efficient data access and synchronization.
 * Outputs the computed values to an array.
 */
__global__ void stencil1D(int *input, int *output) {
    __shared__ int temp[N + 2 * RADIUS]; 
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x + RADIUS; 


    temp[local_tid] = input[tid];
    if (threadIdx.x < RADIUS) {
        temp[local_tid - RADIUS] = input[tid - RADIUS]; 
        temp[local_tid + blockDim.x] = input[tid + blockDim.x]; 
    }
    __syncthreads();

    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++) {
        result += temp[local_tid + offset];
    }

    output[tid] = result;
}

int main() {
    int h_input[N + 2 * RADIUS];
    int h_output[N];


    for (int i = 0; i < N + 2 * RADIUS; i++) {
        h_input[i] = i;
    }

    int *d_input, *d_output;
    cudaMalloc(&d_input, (N + 2 * RADIUS) * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));

    cudaMemcpy(d_input, h_input, (N + 2 * RADIUS) * sizeof(int), cudaMemcpyHostToDevice);

    stencil1D<<<1, N>>>(d_input + RADIUS, d_output);  

    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Stencil output:\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
