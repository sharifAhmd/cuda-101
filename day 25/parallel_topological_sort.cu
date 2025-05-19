#include <iostream>
#define N 6
#define E 7

__global__ void findZeroInDegree(int *in_degree, int *frontier) {
    int tid = threadIdx.x;
    if (in_degree[tid] == 0) {
        frontier[tid] = 1;
        in_degree[tid] = -1; 
    }
}

__global__ void updateInDegrees(int *row_ptr, int *col_idx, int *frontier, int *in_degree) {
    int tid = threadIdx.x;
    if (frontier[tid]) {
        frontier[tid] = 0;
        int start = row_ptr[tid];
        int end = row_ptr[tid + 1];
        for (int i = start; i < end; i++) {
            atomicSub(&in_degree[col_idx[i]], 1);
        }
    }
}

int main() {
    int h_row_ptr[N + 1] = {0, 0, 0, 1, 2, 4, 6};  
    int h_col_idx[E] = {3, 1, 3, 0, 0, 1, 2};      
    int h_in_degree[N] = {2, 2, 1, 1, 0, 0};       
    int h_frontier[N] = {0};
    int h_sorted[N], pos = 0;

    int *d_row_ptr, *d_col_idx, *d_in_degree, *d_frontier;
    cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, E * sizeof(int));
    cudaMalloc(&d_in_degree, N * sizeof(int));
    cudaMalloc(&d_frontier, N * sizeof(int));

    cudaMemcpy(d_row_ptr, h_row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_degree, h_in_degree, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_frontier, 0, N * sizeof(int));

    bool done = false;
    while (!done) {
        findZeroInDegree<<<1, N>>>(d_in_degree, d_frontier);
        cudaMemcpy(h_frontier, d_frontier, N * sizeof(int), cudaMemcpyDeviceToHost);

        done = true;
        for (int i = 0; i < N; i++) {
            if (h_frontier[i]) {
                h_sorted[pos++] = i;
                done = false;
            }
        }

        updateInDegrees<<<1, N>>>(d_row_ptr, d_col_idx, d_frontier, d_in_degree);
    }

    std::cout << "Topological Order: ";
    for (int i = 0; i < pos; i++) std::cout << h_sorted[i] << " ";
    std::cout << "\n";

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_in_degree);
    cudaFree(d_frontier);
    return 0;
}
