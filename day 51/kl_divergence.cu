#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define EPSILON 1e-8  

// https://www.trybackprop.com/blog/2025_05_31_cross_entropy

__global__ void kl_divergence_kernel(const float* P, const float* Q, float* partial, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float p = fmaxf(P[idx], EPSILON);  
        float q = fmaxf(Q[idx], EPSILON); 
        partial[idx] = p * logf(p / q);
    }
}

float kl_divergence(const float* h_P, const float* h_Q, int N) {
    float *d_P, *d_Q, *d_partial;
    float h_partial[N];

    cudaMalloc(&d_P, N * sizeof(float));
    cudaMalloc(&d_Q, N * sizeof(float));
    cudaMalloc(&d_partial, N * sizeof(float));
    cudaMemcpy(d_P, h_P, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    kl_divergence_kernel<<<blocks, threads>>>(d_P, d_Q, d_partial, N);
    cudaMemcpy(h_partial, d_partial, N * sizeof(float), cudaMemcpyDeviceToHost);

    float result = 0.0f;
    for (int i = 0; i < N; ++i)
        result += h_partial[i];

    cudaFree(d_P); cudaFree(d_Q); cudaFree(d_partial);
    return result;
}

int main() {
    const int N = 4;
    float h_P[N] = {0.1, 0.2, 0.3, 0.4};
    float h_Q[N] = {0.25, 0.25, 0.25, 0.25}; 

    float kl = kl_divergence(h_P, h_Q, N);

    printf("KL(P || Q) = %.6f\n", kl);
    return 0;
}
