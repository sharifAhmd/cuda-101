#include <cuda_runtime.h>
#include <stdio.h>

#define N 16 

__global__ void bitonicSortKernel(float* data, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    if (ixj > i) {
        if ((i & k) == 0) {
            if (data[i] > data[ixj]) {
                float temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            if (data[i] < data[ixj]) {
                float temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

void printArray(const char* label, float* data, int n) {
    printf("%s: [", label);
    for (int i = 0; i < n; i++)
        printf(" %.2f", data[i]);
    printf(" ]\n");
}

int main() {
    float h_data[N] = {7, 3, 15, 12, 0, 10, 1, 5, 4, 2, 8, 9, 6, 13, 14, 11};
    float* d_data;

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    printArray("Unsorted", h_data, N);

    dim3 blocks(1);
    dim3 threads(N);
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortKernel<<<blocks, threads>>>(d_data, j, k);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    printArray("Sorted", h_data, N);

    cudaFree(d_data);
    return 0;
}
