#include <iostream>
#define N 8
/*
Remove all zeros from an array using a scan operation.
For example:
Input: 3 0 4 5 0 2 0 1 
Mask: 1 0 1 1 0 1 0 1 
Scan: 1 1 2 3 3 4 4 5 
Compacted Output: 3 4 5 2 1 
*/

__global__ void markValid(int *input, int *mask) {
    int idx = threadIdx.x;
    mask[idx] = (input[idx] != 0) ? 1 : 0;
}

__global__ void inclusiveScan(int *mask, int *scan) {
    __shared__ int temp[N];
    int tid = threadIdx.x;

    temp[tid] = mask[tid];
    __syncthreads();

    for (int stride = 1; stride < N; stride *= 2) {
        int val = 0;
        if (tid >= stride) val = temp[tid - stride];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }
    scan[tid] = temp[tid];
}

__global__ void scatter(int *input, int *mask, int *scan, int *output) {
    int idx = threadIdx.x;
    if (mask[idx]) {
        int pos = scan[idx] - 1;  
        output[pos] = input[idx];
    }
}

int main() {
    int h_input[N] = {3, 0, 4, 5, 0, 2, 0, 1};
    int h_mask[N], h_scan[N], h_output[N];

    int *d_input, *d_mask, *d_scan, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_mask, N * sizeof(int));
    cudaMalloc(&d_scan, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    markValid<<<1, N>>>(d_input, d_mask);
    inclusiveScan<<<1, N>>>(d_mask, d_scan);
    scatter<<<1, N>>>(d_input, d_mask, d_scan, d_output);

    cudaMemcpy(h_mask, d_mask, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scan, d_scan, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Input: ";
    for (int i = 0; i < N; i++) std::cout << h_input[i] << " ";
    std::cout << "\nMask: ";
    for (int i = 0; i < N; i++) std::cout << h_mask[i] << " ";
    std::cout << "\nScan: ";
    for (int i = 0; i < N; i++) std::cout << h_scan[i] << " ";
    std::cout << "\nCompacted Output: ";
    for (int i = 0; i < h_scan[N - 1]; i++) std::cout << h_output[i] << " ";
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_scan);
    cudaFree(d_output);
    return 0;
}
