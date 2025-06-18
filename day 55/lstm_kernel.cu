#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__device__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void lstm_forward(
    const float* x,
    const float* h,
    const float* c,
    const float* Wx,
    const float* Wh,
    const float* b,
    float* next_h,
    float* next_c,
    int batch, int input_dim, int hidden_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < batch && j < hidden_dim) {
        float g[4] = {b[j], b[hidden_dim + j], b[2 * hidden_dim + j], b[3 * hidden_dim + j]};
        for (int d = 0; d < input_dim; ++d)
            for (int k = 0; k < 4; ++k)
                g[k] += x[i * input_dim + d] * Wx[d * 4 * hidden_dim + k * hidden_dim + j];
        for (int d = 0; d < hidden_dim; ++d)
            for (int k = 0; k < 4; ++k)
                g[k] += h[i * hidden_dim + d] * Wh[d * 4 * hidden_dim + k * hidden_dim + j];

        float i_gate = sigmoidf(g[0]);
        float f_gate = sigmoidf(g[1]);
        float o_gate = sigmoidf(g[2]);
        float g_gate = tanhf(g[3]);
        float c_new = f_gate * c[i * hidden_dim + j] + i_gate * g_gate;
        float h_new = o_gate * tanhf(c_new);

        next_c[i * hidden_dim + j] = c_new;
        next_h[i * hidden_dim + j] = h_new;
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
    float h_x[batch * input_dim] = {0.5, 1.0, -0.5, -0.1, 0.2, 0.3};
    float h_h[batch * hidden_dim] = {0.0, 0.0, 0.0, 0.0};
    float h_c[batch * hidden_dim] = {0.0, 0.0, 0.0, 0.0};
    float h_Wx[input_dim * 4 * hidden_dim] = {0.4, -0.5, 0.6, -0.7, 0.1, 0.2, -0.3, 0.4, 0.5, 0.6, -0.1, -0.2, 0.2, -0.3, 0.3, -0.4, 0.1, 0.2, 0.3, 0.4, -0.2, -0.3, 0.2, 0.1};
    float h_Wh[hidden_dim * 4 * hidden_dim] = {0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.2, 0.1, 0.3, -0.1, 0.2, -0.2, 0.2, -0.2, 0.1, 0.3};
    float h_b[4 * hidden_dim] = {0.1, 0.2, 0.0, -0.1, 0.0, 0.1, -0.2, 0.2};
    float h_next_h[batch * hidden_dim], h_next_c[batch * hidden_dim];

    float *d_x, *d_h, *d_c, *d_Wx, *d_Wh, *d_b, *d_next_h, *d_next_c;
    cudaMalloc(&d_x, batch * input_dim * sizeof(float));
    cudaMalloc(&d_h, batch * hidden_dim * sizeof(float));
    cudaMalloc(&d_c, batch * hidden_dim * sizeof(float));
    cudaMalloc(&d_Wx, input_dim * 4 * hidden_dim * sizeof(float));
    cudaMalloc(&d_Wh, hidden_dim * 4 * hidden_dim * sizeof(float));
    cudaMalloc(&d_b, 4 * hidden_dim * sizeof(float));
    cudaMalloc(&d_next_h, batch * hidden_dim * sizeof(float));
    cudaMalloc(&d_next_c, batch * hidden_dim * sizeof(float));

    cudaMemcpy(d_x, h_x, batch * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h_h, batch * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, batch * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wx, h_Wx, input_dim * 4 * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wh, h_Wh, hidden_dim * 4 * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 4 * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(2, 2);
    dim3 grid((batch + block.x - 1) / block.x, (hidden_dim + block.y - 1) / block.y);
    lstm_forward<<<grid, block>>>(d_x, d_h, d_c, d_Wx, d_Wh, d_b, d_next_h, d_next_c, batch, input_dim, hidden_dim);

    cudaMemcpy(h_next_h, d_next_h, batch * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_next_c, d_next_c, batch * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);

    print_matrix("next_h", h_next_h, batch, hidden_dim);
    print_matrix("next_c", h_next_c, batch, hidden_dim);

    cudaFree(d_x); cudaFree(d_h); cudaFree(d_c); cudaFree(d_Wx); cudaFree(d_Wh); cudaFree(d_b); cudaFree(d_next_h); cudaFree(d_next_c);
    return 0;
}
