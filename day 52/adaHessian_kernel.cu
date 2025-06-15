#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void adahessian_update(
    float* param,        // [n] parameters to update
    const float* grad,   // [n] gradients
    const float* hess_diag, // [n] diagonal Hessian estimates
    float* exp_avg,      // [n] running mean of gradients (momentum)
    float* exp_hessian_sq, // [n] running mean of squared hessian diagonals
    float lr,
    float beta1,
    float beta2,
    float eps,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        exp_avg[i] = beta1 * exp_avg[i] + (1 - beta1) * grad[i];
        exp_hessian_sq[i] = beta2 * exp_hessian_sq[i] + (1 - beta2) * hess_diag[i] * hess_diag[i];
        float denom = sqrtf(exp_hessian_sq[i]) + eps;
        param[i] -= lr * exp_avg[i] / denom;
    }
}

int main() {
    const int n = 8;
    float h_param[n]        = {0.2, 0.4, 0.1, -0.3, 0.5, 0.6, 0.7, -0.1};
    float h_grad[n]         = {0.1, -0.2, 0.0, 0.3, -0.1, 0.2, -0.3, 0.0};
    float h_hess_diag[n]    = {1.0, 1.2, 1.1, 1.5, 0.9, 1.3, 1.0, 0.8};
    float h_exp_avg[n]      = {0.0};
    float h_exp_hessian_sq[n] = {0.0};

    float *d_param, *d_grad, *d_hess_diag, *d_exp_avg, *d_exp_hessian_sq;
    cudaMalloc(&d_param, n * sizeof(float));
    cudaMalloc(&d_grad, n * sizeof(float));
    cudaMalloc(&d_hess_diag, n * sizeof(float));
    cudaMalloc(&d_exp_avg, n * sizeof(float));
    cudaMalloc(&d_exp_hessian_sq, n * sizeof(float));

    cudaMemcpy(d_param, h_param, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad, h_grad, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hess_diag, h_hess_diag, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_exp_avg, h_exp_avg, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_exp_hessian_sq, h_exp_hessian_sq, n * sizeof(float), cudaMemcpyHostToDevice);

    float lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-4;
    int threads = 256, blocks = (n + threads - 1) / threads;
    adahessian_update<<<blocks, threads>>>(
        d_param, d_grad, d_hess_diag, d_exp_avg, d_exp_hessian_sq,
        lr, beta1, beta2, eps, n
    );
    cudaMemcpy(h_param, d_param, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Updated parameters:\n[");
    for (int i = 0; i < n; ++i)
        printf(" %.6f", h_param[i]);
    printf(" ]\n");

    cudaFree(d_param); cudaFree(d_grad); cudaFree(d_hess_diag); cudaFree(d_exp_avg); cudaFree(d_exp_hessian_sq);
    return 0;
}
