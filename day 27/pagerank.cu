#include <iostream>
#include <cmath>
#define N 4
#define E 6
#define DAMPING 0.85
#define EPSILON 1e-5
#define MAX_ITER 100

__global__ void pagerank_kernel(int *row_ptr, int *col_idx, float *ranks, float *new_ranks, int *out_degree) {
    int i = threadIdx.x;

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        int start = row_ptr[j];
        int end = row_ptr[j + 1];
        for (int k = start; k < end; k++) {
            if (col_idx[k] == i && out_degree[j] > 0) {
                sum += ranks[j] / out_degree[j];
            }
        }
    }

    new_ranks[i] = (1.0f - DAMPING) / N + DAMPING * sum;
}

__global__ void calculate_error(float *old_ranks, float *new_ranks, float *error) {
    int i = threadIdx.x;
    float diff = fabsf(old_ranks[i] - new_ranks[i]);
    atomicAdd(error, diff);
}

int main() {

    int h_row_ptr[N + 1] = {0, 2, 3, 4, 6};
    int h_col_idx[E] = {1, 2, 2, 0, 2, 0};
    int h_out_degree[N] = {2, 1, 1, 2};

    float h_ranks[N], h_new_ranks[N], h_error;
    for (int i = 0; i < N; i++) h_ranks[i] = 1.0f / N;

    int *d_row_ptr, *d_col_idx, *d_out_degree;
    float *d_ranks, *d_new_ranks, *d_error;

    cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, E * sizeof(int));
    cudaMalloc(&d_out_degree, N * sizeof(int));
    cudaMalloc(&d_ranks, N * sizeof(float));
    cudaMalloc(&d_new_ranks, N * sizeof(float));
    cudaMalloc(&d_error, sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_degree, h_out_degree, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ranks, h_ranks, N * sizeof(float), cudaMemcpyHostToDevice);

    int iter = 0;
    do {
        h_error = 0.0f;
        cudaMemcpy(d_error, &h_error, sizeof(float), cudaMemcpyHostToDevice);

        pagerank_kernel<<<1, N>>>(d_row_ptr, d_col_idx, d_ranks, d_new_ranks, d_out_degree);
        calculate_error<<<1, N>>>(d_ranks, d_new_ranks, d_error);

        cudaMemcpy(&h_error, d_error, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_ranks, d_new_ranks, N * sizeof(float), cudaMemcpyDeviceToDevice);

        iter++;
    } while (h_error > EPSILON && iter < MAX_ITER);

    cudaMemcpy(h_ranks, d_ranks, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "PageRank after " << iter << " iterations:\n";
    for (int i = 0; i < N; i++) {
        std::cout << "Node " << i << ": " << h_ranks[i] << "\n";
    }

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_out_degree);
    cudaFree(d_ranks);
    cudaFree(d_new_ranks);
    cudaFree(d_error);

    return 0;
}
