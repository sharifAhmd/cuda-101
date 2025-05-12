#include <iostream>
#define N 12

//removes all negative numbers from an array using parallel stream compaction
__global__ void markValid(int *input, int *mask) {
    int idx = threadIdx.x;
    mask[idx] = (input[idx] >= 0) ? 1 : 0;
}

__global__ void exclusiveScan(int *mask, int *scan) {
    __shared__ int temp[N];
    int tid = threadIdx.x;

    if (tid == 0) temp[0] = 0;
    else temp[tid] = mask[tid - 1];
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
        int pos = scan[idx];
        output[pos] = input[idx];
    }
}

int main() {
    int h_input[N] = {5, -1, 3, 0, -4, 8, 7, -3, 2, -9, 6, 1};
    int h_mask[N], h_scan[N], h_output[N];

    int *d_input, *d_mask, *d_scan, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_mask, N * sizeof(int));
    cudaMalloc(&d_scan, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    markValid<<<1, N>>>(d_input, d_mask);
    exclusiveScan<<<1, N>>>(d_mask, d_scan);
    scatter<<<1, N>>>(d_input, d_mask, d_scan, d_output);

    cudaMemcpy(h_mask, d_mask, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scan, d_scan, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Compacted Output (non-negatives): ";
    for (int i = 0; i < h_scan[N - 1] + h_mask[N - 1]; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << "\n";

    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_scan);
    cudaFree(d_output);
    return 0;
}
