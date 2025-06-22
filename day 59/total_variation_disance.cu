#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void tv_distance_kernel(const float* P, const float* Q, float* partial, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        partial[idx] = fabsf(P[idx] - Q[idx]);
    }
}

float tv_distance(const float* h_P, const float* h_Q, int N) {
    float *d_P, *d_Q, *d_partial;
    float* h_partial = (float*)malloc(N * sizeof(float));
    cudaMalloc(&d_P, N * sizeof(float));
    cudaMalloc(&d_Q, N * sizeof(float));
    cudaMalloc(&d_partial, N * sizeof(float));
    cudaMemcpy(d_P, h_P, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tv_distance_kernel<<<blocks, threads>>>(d_P, d_Q, d_partial, N);
    cudaMemcpy(h_partial, d_partial, N * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < N; ++i)
        sum += h_partial[i];

    cudaFree(d_P); cudaFree(d_Q); cudaFree(d_partial); free(h_partial);
    return 0.5f * sum;
}

int main() {
    const int N = 5;
    float P[N] = {0.1f, 0.2f, 0.3f, 0.3f, 0.1f};
    float Q[N] = {0.2f, 0.1f, 0.4f, 0.2f, 0.1f};
    float tv = tv_distance(P, Q, N);
    printf("TV distance = %.6f\n", tv);
    return 0;
}
