#include <iostream>
#define N 8

__global__ void prefixSum(int *input, int *output) {
    /**
     * This kernel computes the prefix sum of an array using shared memory
     * Each thread loads one element from global memory to shared memory
     * Then, it performs the prefix sum using the values in shared memory
     * Finally, it writes the result back to global memory
     */
    __shared__ int temp[N];  
    int tid = threadIdx.x;

    temp[tid] = input[tid];  
    __syncthreads();


    for (int stride = 1; stride < N; stride *= 2) {
        int val = 0;
        if (tid >= stride) {
            val = temp[tid - stride];
        }
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    output[tid] = temp[tid];  
}

int main() {
    int h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int h_output[N];

    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    prefixSum<<<1, N>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Prefix Sum Output:\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
