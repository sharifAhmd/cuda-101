#include <iostream>
#define N 8

__global__ void reduceSum(int *input, int *output) {
    __shared__ int shared_data[N];

    int tid = threadIdx.x;
    shared_data[tid] = input[tid];
    __syncthreads();

    // Reduction loop
    for (int stride = N / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = shared_data[0];
    }
}

int main() {
    int h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int h_output;

    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    reduceSum<<<1, N>>>(d_input, d_output);

    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sum = " << h_output << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
