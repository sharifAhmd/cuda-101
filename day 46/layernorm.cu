#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>


__global__ void layernorm_kernel(const float* input, float* output, int rows, int cols,
                                 const float* gamma, const float* beta, float eps) {
    int row = blockIdx.x;
    if (row < rows) {
        float mean = 0.0f;
        for (int i = 0; i < cols; ++i)
            mean += input[row * cols + i];
        mean /= cols;

        float var = 0.0f;
        for (int i = 0; i < cols; ++i) {
            float diff = input[row * cols + i] - mean;
            var += diff * diff;
        }
        var /= cols;

        for (int i = 0; i < cols; ++i) {
            float norm = (input[row * cols + i] - mean) / sqrtf(var + eps);
            output[row * cols + i] = gamma[i] * norm + beta[i];
        }
    }
}

void print_array(const char* name, const float* arr, int n) {
    printf("%s: [", name);
    for (int i = 0; i < n; ++i) printf(" %.4f", arr[i]);
    printf(" ]\n");
}

int main() {

    const int rows = 2, cols = 4;
    float h_mat[rows * cols] = { 1, 2, 3, 4, 10, 20, 30, 40 };
    float h_gamma[cols] = { 1, 1, 1, 1 }; 
    float h_beta[cols]  = { 0, 0, 0, 0 };  
    float h_normed[rows * cols];

    float *d_mat, *d_gamma, *d_beta, *d_normed;
    cudaMalloc(&d_mat, rows * cols * sizeof(float));
    cudaMalloc(&d_gamma, cols * sizeof(float));
    cudaMalloc(&d_beta, cols * sizeof(float));
    cudaMalloc(&d_normed, rows * cols * sizeof(float));
    cudaMemcpy(d_mat, h_mat, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, cols * sizeof(float), cudaMemcpyHostToDevice);

    layernorm_kernel<<<rows, 1>>>(d_mat, d_normed, rows, cols, d_gamma, d_beta, 1e-5f);
    cudaMemcpy(h_normed, d_normed, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    printf("LayerNorm output:\n");
    for (int r = 0; r < rows; ++r) {
        printf("Row %d: [", r);
        for (int c = 0; c < cols; ++c)
            printf(" %.4f", h_normed[r * cols + c]);
        printf(" ]\n");
    }

    cudaFree(d_mat); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_normed);

    return 0;
}
