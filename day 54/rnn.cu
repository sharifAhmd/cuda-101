#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__device__ float tanhf_cuda(float x) {
    return tanhf(x);
}

__global__ void rnn_forward(
    const float* x,
    const float* h,
    const float* Wx,
    const float* Wh,
    const float* b,
    float* next_h,
    int batch, int input_dim, int hidden_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < batch && j < hidden_dim) {
        float val = b[j];
        for (int d = 0; d < input_dim; ++d)
            val += x[i * input_dim + d] * Wx[d * hidden_dim + j];
        for (int d = 0; d < hidden_dim; ++d)
            val += h[i * hidden_dim + d] * Wh[d * hidden_dim + j];
        next_h[i * hidden_dim + j] = tanhf_cuda(val);
    }
}

void print_matrix(const char* label, float* data, int rows, int cols) {
    printf("%s:\n", label);
    for (int r = 0; r < rows; ++r) {
        printf("[");
        for (int c = 0; c < cols; ++c)
            printf(" %.4f", data[r * cols + c]);
        printf(" ]\n");
    }
}

int main() {
    const int batch = 2, input_dim = 3, hidden_dim = 2;
    float h_x[batch * input_dim] = {0.1, 0.2, 0.3, -0.1, 0.4, 0.5};
    float h_h[batch * hidden_dim] = {0.0, 0.0, 0.0, 0.0};
    float h_Wx[input_dim * hidden_dim] = {0.2, -0.3, 0.4, 0.1, -0.2, 0.5};
    float h_Wh[hidden_dim * hidden_dim] = {0.3, -0.4, 0.5, 0.6};
    float h_b[hidden_dim] = {0.1, -0.2};
    float h_next_h[batch * hidden_dim];

    float *d_x, *d_h, *d_Wx, *d_Wh, *d_b, *d_next_h;
    cudaMalloc(&d_x, batch * input_dim * sizeof(float));
    cudaMalloc(&d_h, batch * hidden_dim * sizeof(float));
    cudaMalloc(&d_Wx, input_dim * hidden_dim * sizeof(float));
    cudaMalloc(&d_Wh, hidden_dim * hidden_dim * sizeof(float));
    cudaMalloc(&d_b, hidden_dim * sizeof(float));
    cudaMalloc(&d_next_h, batch * hidden_dim * sizeof(float));

    cudaMemcpy(d_x, h_x, batch * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h_h, batch * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wx, h_Wx, input_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wh, h_Wh, hidden_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, hidden_dim * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(2, 2);
    dim3 grid((batch + block.x - 1) / block.x, (hidden_dim + block.y - 1) / block.y);
    rnn_forward<<<grid, block>>>(d_x, d_h, d_Wx, d_Wh, d_b, d_next_h, batch, input_dim, hidden_dim);

    cudaMemcpy(h_next_h, d_next_h, batch * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);

    print_matrix("next_h", h_next_h, batch, hidden_dim);

    cudaFree(d_x); cudaFree(d_h); cudaFree(d_Wx); cudaFree(d_Wh); cudaFree(d_b); cudaFree(d_next_h);
    return 0;
}
