#include <iostream>
#define N 8

/**
 * Performs a prefix sum (scan) operation on the input array.
 * The kernel uses shared memory for efficient data access and synchronization.
 * The prefix sum is computed in-place, and the result is stored in the output array.
 */

__global__ void prefixSum(int *input, int *output) {
    __shared__ int temp[N];

    int tid = threadIdx.x;

    temp[tid] = input[tid];
    __syncthreads();

    for (int stride = 1; stride < N; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < N) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    for (int stride = N / 4; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < N) {
            temp[index + stride] += temp[index];
        }
        __syncthreads();
    }

    output[tid] = temp[tid];
}

int main() {
    int h_input[N] = {3, 1, 7, 0, 4, 1, 6, 3};
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
