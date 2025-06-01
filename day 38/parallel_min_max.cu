#include <iostream>
#include <limits>
#include <cuda_runtime.h>

#define N 1024
#define TPB 256 

__global__ void findMinMax(const float* input, float* min_out, float* max_out) {
    __shared__ float s_min[TPB];
    __shared__ float s_max[TPB];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float val = input[idx];
    s_min[tid] = val;
    s_max[tid] = val;
    __syncthreads();


    for (int stride = TPB / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + stride]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        min_out[blockIdx.x] = s_min[0];
        max_out[blockIdx.x] = s_max[0];
    }
}

int main() {
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }

    float *d_data, *d_min, *d_max;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = N / TPB;
    cudaMalloc(&d_min, blocks * sizeof(float));
    cudaMalloc(&d_max, blocks * sizeof(float));

    findMinMax<<<blocks, TPB>>>(d_data, d_min, d_max);
    cudaDeviceSynchronize();

    float* h_min = new float[blocks];
    float* h_max = new float[blocks];
    cudaMemcpy(h_min, d_min, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max, d_max, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float final_min = h_min[0];
    float final_max = h_max[0];
    for (int i = 1; i < blocks; i++) {
        final_min = std::min(final_min, h_min[i]);
        final_max = std::max(final_max, h_max[i]);
    }

    std::cout << "Min: " << final_min << "\n";
    std::cout << "Max: " << final_max << "\n";

    cudaFree(d_data);
    cudaFree(d_min);
    cudaFree(d_max);
    delete[] h_data;
    delete[] h_min;
    delete[] h_max;

    return 0;
}
