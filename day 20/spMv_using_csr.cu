#include <iostream>

#define NUM_ROWS 4
#define NUM_NONZERO 9
/*
 Sparse Matrix-Vector Multiplication
*/
__global__ void spmv_csr_kernel(
    int *row_ptr, int *col_idx, float *values,
    float *x, float *y, int num_rows) {

    int row = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < num_rows) {
        float dot = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        for (int i = row_start; i < row_end; i++) {
            dot += values[i] * x[col_idx[i]];
        }
        y[row] = dot;
    }
}

int main() {

    int h_row_ptr[NUM_ROWS + 1] = {0, 1, 3, 6, 7};
    int h_col_idx[NUM_NONZERO] = {0, 1, 2, 0, 2, 3, 3};
    float h_values[NUM_NONZERO] = {10, 20, 30, 40, 50, 60, 70};
    float h_x[4] = {1, 2, 3, 4};
    float h_y[NUM_ROWS] = {0};

    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_row_ptr, (NUM_ROWS + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, NUM_NONZERO * sizeof(int));
    cudaMalloc(&d_values, NUM_NONZERO * sizeof(float));
    cudaMalloc(&d_x, 4 * sizeof(float));
    cudaMalloc(&d_y, NUM_ROWS * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr, (NUM_ROWS + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, NUM_NONZERO * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, NUM_NONZERO * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, 4 * sizeof(float), cudaMemcpyHostToDevice);

    spmv_csr_kernel<<<1, NUM_ROWS>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, NUM_ROWS);

    cudaMemcpy(h_y, d_y, NUM_ROWS * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result y = A * x:\n";
    for (int i = 0; i < NUM_ROWS; i++) {
        std::cout << h_y[i] << " ";
    }
    std::cout << "\n";

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
