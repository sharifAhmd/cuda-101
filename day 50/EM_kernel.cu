#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void estep_kernel(const float* x, int N, const float* means, const float* vars, const float* weights, int K, float* resp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        float likelihoods[8]; // Max K = 8 for demo
        for (int k = 0; k < K; ++k) {
            float diff = x[i] - means[k];
            float var = vars[k];
            float w = weights[k];
            float l = w * expf(-0.5f * diff * diff / var) / sqrtf(2 * M_PI * var);
            likelihoods[k] = l;
            sum += l;
        }
        for (int k = 0; k < K; ++k) {
            resp[i * K + k] = likelihoods[k] / sum; // Normalize to get responsibilities
        }
    }
}

void print_matrix(const char* name, float* mat, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; ++i) {
        printf("[");
        for (int j = 0; j < cols; ++j)
            printf(" %.4f", mat[i * cols + j]);
        printf(" ]\n");
    }
}

int main() {
    const int N = 6, K = 2;
    float h_x[N] = {1.0, 1.2, 0.8, 4.0, 4.2, 3.8};
    float h_means[K] = {1.0, 4.0};
    float h_vars[K]  = {0.1, 0.1};
    float h_weights[K] = {0.5, 0.5};
    float h_resp[N * K];

    float *d_x, *d_means, *d_vars, *d_weights, *d_resp;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_means, K * sizeof(float));
    cudaMalloc(&d_vars, K * sizeof(float));
    cudaMalloc(&d_weights, K * sizeof(float));
    cudaMalloc(&d_resp, N * K * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, h_means, K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vars, h_vars, K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, K * sizeof(float), cudaMemcpyHostToDevice);

    estep_kernel<<<1, N>>>(d_x, N, d_means, d_vars, d_weights, K, d_resp);

    cudaMemcpy(h_resp, d_resp, N * K * sizeof(float), cudaMemcpyDeviceToHost);

    print_matrix("Responsibilities (E-step output)", h_resp, N, K);

    cudaFree(d_x); cudaFree(d_means); cudaFree(d_vars); cudaFree(d_weights); cudaFree(d_resp);
    return 0;
}
