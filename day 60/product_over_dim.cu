#include <cuda_runtime.h>
#include <stdio.h>

__global__ void row_product(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float prod = 1.0f;
        for (int c = 0; c < cols; ++c)
            prod *= input[row * cols + c];
        output[row] = prod;
    }
}

void print_array(const char* label, const float* arr, int n) {
    printf("%s: [", label);
    for (int i = 0; i < n; ++i)
        printf(" %.6f", arr[i]);
    printf(" ]\n");
}

int main() {
    const int rows = 4, cols = 3;
    float h_in[rows * cols] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        -1, 2, -3
    };
    float h_out[rows];

    float *d_in, *d_out;
    cudaMalloc(&d_in, rows * cols * sizeof(float));
    cudaMalloc(&d_out, rows * sizeof(float));
    cudaMemcpy(d_in, h_in, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128, blocks = (rows + threads - 1) / threads;
    row_product<<<blocks, threads>>>(d_in, d_out, rows, cols);
    cudaMemcpy(h_out, d_out, rows * sizeof(float), cudaMemcpyDeviceToHost);

    print_array("Input row products", h_out, rows);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
