#include <iostream>

#define N 16       
#define BINS 4     

__global__ void buildHistogram(int *input, int *hist) {
    int tid = threadIdx.x;
    atomicAdd(&hist[input[tid]], 1);  
}

__global__ void inclusiveScan(int *input, int *output) {
    __shared__ int temp[BINS];
    int tid = threadIdx.x;

    temp[tid] = input[tid];
    __syncthreads();

    for (int stride = 1; stride < BINS; stride *= 2) {
        int val = 0;
        if (tid >= stride) val = temp[tid - stride];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    output[tid] = temp[tid];
}

int main() {
    int h_input[N] = {0, 1, 1, 2, 3, 2, 0, 0, 1, 2, 2, 3, 3, 1, 0, 0};
    int h_hist[BINS] = {0};
    int h_cdf[BINS] = {0};

    int *d_input, *d_hist, *d_cdf;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_hist, BINS * sizeof(int));
    cudaMalloc(&d_cdf, BINS * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, BINS * sizeof(int));

    buildHistogram<<<1, N>>>(d_input, d_hist);
    inclusiveScan<<<1, BINS>>>(d_hist, d_cdf);

    cudaMemcpy(h_hist, d_hist, BINS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cdf, d_cdf, BINS * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Histogram:\n";
    for (int i = 0; i < BINS; i++) {
        std::cout << "Bin[" << i << "] = " << h_hist[i] << "\n";
    }

    std::cout << "\nCDF (Prefix Sum):\n";
    for (int i = 0; i < BINS; i++) {
        std::cout << "CDF[" << i << "] = " << h_cdf[i] << "\n";
    }

    cudaFree(d_input);
    cudaFree(d_hist);
    cudaFree(d_cdf);
    return 0;
}
