#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <math.h>

#if __CUDACC_VER_MAJOR__ >= 12
#include <cuda_fp8.h>
#endif

#define N 1024
#define BLOCK 16

__global__ void matmul_fp32(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (row < n && col < n) {
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

__global__ void matmul_fp16(const half* A, const half* B, half* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    half sum = __float2half(0.f);
    if (row < n && col < n) {
        for (int k = 0; k < n; ++k)
            sum = __hadd(sum, __hmul(A[row * n + k], B[k * n + col]));
        C[row * n + col] = sum;
    }
}

#if __CUDACC_VER_MAJOR__ >= 12
__global__ void matmul_fp8(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, __nv_fp8_e4m3* C, int n, float scaleA, float scaleB, float scaleC) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.f;
    if (row < n && col < n) {
        for (int k = 0; k < n; ++k) {
            float a = __nv_fp8_e4m3_to_fp32(A[row * n + k], scaleA);
            float b = __nv_fp8_e4m3_to_fp32(B[k * n + col], scaleB);
            sum += a * b;
        }
        C[row * n + col] = __nv_fp32_to_fp8_e4m3(sum, scaleC);
    }
}
#endif

void fill_rand(float* mat, int n) {
    for (int i = 0; i < n * n; ++i)
        mat[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

float compare(const float* ref, const float* pred, int n) {
    float max_rel_err = 0;
    for (int i = 0; i < n * n; ++i) {
        float err = fabsf(ref[i] - pred[i]);
        float rel = (fabsf(ref[i]) > 1e-6) ? err / fabsf(ref[i]) : err;
        if (rel > max_rel_err) max_rel_err = rel;
    }
    return max_rel_err;
}

int main() {
    srand(time(0));
    float *h_A = (float*)malloc(N*N*sizeof(float));
    float *h_B = (float*)malloc(N*N*sizeof(float));
    float *h_C = (float*)malloc(N*N*sizeof(float));
    float *h_C_fp16 = (float*)malloc(N*N*sizeof(float));
    float *h_C_fp8 = (float*)malloc(N*N*sizeof(float));

    fill_rand(h_A, N);
    fill_rand(h_B, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*N*sizeof(float));
    cudaMalloc(&d_B, N*N*sizeof(float));
    cudaMalloc(&d_C, N*N*sizeof(float));
    cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK, BLOCK), grid((N+BLOCK-1)/BLOCK, (N+BLOCK-1)/BLOCK);

    cudaEvent_t start, stop;
    float time_fp32, time_fp16, time_fp8;

    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul_fp32<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_fp32, start, stop);

    cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    half *d_A16, *d_B16, *d_C16;
    cudaMalloc(&d_A16, N*N*sizeof(half));
    cudaMalloc(&d_B16, N*N*sizeof(half));
    cudaMalloc(&d_C16, N*N*sizeof(half));
    half* h_A16 = (half*)malloc(N*N*sizeof(half));
    half* h_B16 = (half*)malloc(N*N*sizeof(half));
    for (int i = 0; i < N*N; ++i) {
        h_A16[i] = __float2half(h_A[i]);
        h_B16[i] = __float2half(h_B[i]);
    }
    cudaMemcpy(d_A16, h_A16, N*N*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B16, h_B16, N*N*sizeof(half), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    matmul_fp16<<<grid, block>>>(d_A16, d_B16, d_C16, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_fp16, start, stop);

    half* h_C16 = (half*)malloc(N*N*sizeof(half));
    cudaMemcpy(h_C16, d_C16, N*N*sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N*N; ++i) h_C_fp16[i] = __half2float(h_C16[i]);
    float rel_fp16 = compare(h_C, h_C_fp16, N);

#if __CUDACC_VER_MAJOR__ >= 12
    __nv_fp8_e4m3 *d_A8, *d_B8, *d_C8;
    cudaMalloc(&d_A8, N*N*sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_B8, N*N*sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_C8, N*N*sizeof(__nv_fp8_e4m3));
    __nv_fp8_e4m3* h_A8 = (__nv_fp8_e4m3*)malloc(N*N*sizeof(__nv_fp8_e4m3));
    __nv_fp8_e4m3* h_B8 = (__nv_fp8_e4m3*)malloc(N*N*sizeof(__nv_fp8_e4m3));
    float scaleA = 1.0f, scaleB = 1.0f, scaleC = 1.0f;
    for (int i = 0; i < N*N; ++i) {
        h_A8[i] = __nv_fp32_to_fp8_e4m3(h_A[i], scaleA);
        h_B8[i] = __nv_fp32_to_fp8_e4m3(h_B[i], scaleB);
    }
    cudaMemcpy(d_A8, h_A8, N*N*sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B8, h_B8, N*N*sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    matmul_fp8<<<grid, block>>>(d_A8, d_B8, d_C8, N, scaleA, scaleB, scaleC);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_fp8, start, stop);

    __nv_fp8_e4m3* h_C8 = (__nv_fp8_e4m3*)malloc(N*N*sizeof(__nv_fp8_e4m3));
    cudaMemcpy(h_C8, d_C8, N*N*sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N*N; ++i) h_C_fp8[i] = __nv_fp8_e4m3_to_fp32(h_C8[i], scaleC);
    float rel_fp8 = compare(h_C, h_C_fp8, N);
#endif

    printf("FP32 time: %.3f ms\n", time_fp32);
    printf("FP16 time: %.3f ms\n", time_fp16);
    printf("FP16 max relative error: %.6f\n", rel_fp16);
#if __CUDACC_VER_MAJOR__ >= 12
    printf("FP8 time:  %.3f ms\n", time_fp8);
    printf("FP8 max relative error: %.6f\n", rel_fp8);
#else
    printf("FP8 not supported (requires CUDA 12+ and Hopper/Ada GPU).\n");
#endif

    free(h_A); free(h_B); free(h_C); free(h_C_fp16); free(h_C_fp8);
    free(h_A16); free(h_B16); free(h_C16);
#if __CUDACC_VER_MAJOR__ >= 12
    free(h_A8); free(h_B8); free(h_C8);
    cudaFree(d_A8); cudaFree(d_B8); cudaFree(d_C8);
#endif
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A16); cudaFree(d_B16); cudaFree(d_C16);
    return 0;
}
