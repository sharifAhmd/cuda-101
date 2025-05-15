#include <iostream>

#define NNZ 6  
/*
It transposes a sparse matrix in Coordinate List(COO) format by swapping row and column indices of 
all non-zero entries. Each thread processes one element, demonstrating indirect memory access and 
gather-like behavior.
*/
__global__ void transposeCOO(int *row_idx, int *col_idx, float *values,
                             int *t_row_idx, int *t_col_idx, float *t_values) {
    int i = threadIdx.x;

    t_row_idx[i] = col_idx[i];
    t_col_idx[i] = row_idx[i];
    t_values[i]   = values[i];
}

int main() {

    int h_row_idx[NNZ] = {0, 0, 2, 2, 2, 0};  
    int h_col_idx[NNZ] = {0, 2, 0, 1, 1, 1}; 
    float h_values[NNZ] = {5, 8, 3, 6, 1, 4}; 

    int h_t_row_idx[NNZ], h_t_col_idx[NNZ];
    float h_t_values[NNZ];

    int *d_row_idx, *d_col_idx, *d_t_row_idx, *d_t_col_idx;
    float *d_values, *d_t_values;

    cudaMalloc(&d_row_idx, NNZ * sizeof(int));
    cudaMalloc(&d_col_idx, NNZ * sizeof(int));
    cudaMalloc(&d_values, NNZ * sizeof(float));
    cudaMalloc(&d_t_row_idx, NNZ * sizeof(int));
    cudaMalloc(&d_t_col_idx, NNZ * sizeof(int));
    cudaMalloc(&d_t_values, NNZ * sizeof(float));

    cudaMemcpy(d_row_idx, h_row_idx, NNZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, NNZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, NNZ * sizeof(float), cudaMemcpyHostToDevice);

    transposeCOO<<<1, NNZ>>>(d_row_idx, d_col_idx, d_values,
                              d_t_row_idx, d_t_col_idx, d_t_values);

    cudaMemcpy(h_t_row_idx, d_t_row_idx, NNZ * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_t_col_idx, d_t_col_idx, NNZ * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_t_values, d_t_values, NNZ * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Transposed COO Entries:\n";
    for (int i = 0; i < NNZ; i++) {
        std::cout << "Row: " << h_t_row_idx[i]
                  << ", Col: " << h_t_col_idx[i]
                  << ", Val: " << h_t_values[i] << "\n";
    }

    cudaFree(d_row_idx);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_t_row_idx);
    cudaFree(d_t_col_idx);
    cudaFree(d_t_values);

    return 0;
}
