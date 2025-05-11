#include <iostream>
#define N 8


/*
It assigns unique indices to valid (non-zero) flags using an exclusive prefix sum (scan).
Each thread checks if its flag is 1 and, if so, uses the scanned result to compute its position in a compacted array.
For example:
Flags:    0 1 0 1 1 0 1 0
Indexes:  0 0 1 1 2 3 3 4
Assigned: _ 0 _ 1 2 _ 3 _
*/

__global__ void exclusiveScan(int *input, int *output) {
    __shared__ int temp[N];
    int tid = threadIdx.x;

    if (tid == 0) temp[0] = 0;
    else temp[tid] = input[tid - 1];

    __syncthreads();

    for (int stride = 1; stride < N; stride *= 2) {
        int val = 0;
        if (tid >= stride) val = temp[tid - stride];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    output[tid] = temp[tid];
}

__global__ void assignIndices(int *flags, int *indices, int *output) {
    int tid = threadIdx.x;
    if (flags[tid] == 1) {
        output[tid] = indices[tid];
    } else {
        output[tid] = -1; 
    }
}

int main() {
    int h_flags[N] = {0, 1, 0, 1, 1, 0, 1, 0};
    int h_scan[N], h_output[N];

    int *d_flags, *d_scan, *d_output;
    cudaMalloc(&d_flags, N * sizeof(int));
    cudaMalloc(&d_scan, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));

    cudaMemcpy(d_flags, h_flags, N * sizeof(int), cudaMemcpyHostToDevice);

    exclusiveScan<<<1, N>>>(d_flags, d_scan);
    assignIndices<<<1, N>>>(d_flags, d_scan, d_output);

    cudaMemcpy(h_scan, d_scan, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Flags:    ";
    for (int i = 0; i < N; i++) std::cout << h_flags[i] << " ";
    std::cout << "\nIndexes:  ";
    for (int i = 0; i < N; i++) std::cout << h_scan[i] << " ";
    std::cout << "\nAssigned: ";
    for (int i = 0; i < N; i++) {
        if (h_output[i] != -1) std::cout << h_output[i] << " ";
        else std::cout << "_ ";
    }
    std::cout << std::endl;

    cudaFree(d_flags);
    cudaFree(d_scan);
    cudaFree(d_output);
    return 0;
}
