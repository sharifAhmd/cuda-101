#include <iostream>
#include <cuda_runtime.h>

#define M 128  
#define K 64   
#define N 32  

__global__ void dense_layer(const float* X, const float* W, float* Y, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += X[row * K + k] * W[col * K + k];  
        }
        Y[row * N + col] = sum;
    }
}

int main() {
    float *h_X = new float[M * K];
    float *h_W = new float[N * K];
    float *h_Y = new float[M * N];

    // Initialize X and W with simple data
    for (int i = 0; i < M * K; ++i) h_X[i] = 1.0f;
    for (int i = 0; i < N * K; ++i) h_W[i] = 0.5f;

    float *d_X, *d_W, *d_Y;
    cudaMalloc(&d_X, M * K * sizeof(float));
    cudaMalloc(&d_W, N * K * sizeof(float));
    cudaMalloc(&d_Y, M * N * sizeof(float));

    cudaMemcpy(d_X, h_X, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    dense_layer<<<blocks, threads>>>(d_X, d_W, d_Y, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_Y, d_Y, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "First 5 output values:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << h_Y[i] << " ";
    }
    std::cout << "\n";

    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_Y);
    delete[] h_X;
    delete[] h_W;
    delete[] h_Y;

    return 0;
}

