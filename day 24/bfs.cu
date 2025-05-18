#include <iostream>
#define N 7  
#define E 10

__global__ void bfs_kernel(int *row_ptr, int *col_idx, int *visited, int *frontier, int *next_frontier, int *done) {
    int tid = threadIdx.x;

    if (frontier[tid]) {
        frontier[tid] = 0;  
        int start = row_ptr[tid];
        int end = row_ptr[tid + 1];

        for (int i = start; i < end; i++) {
            int neighbor = col_idx[i];
            if (!visited[neighbor]) {
                visited[neighbor] = 1;
                next_frontier[neighbor] = 1;
                *done = 0;  
            }
        }
    }
}

int main() {

    int h_row_ptr[N + 1] = {0, 2, 5, 8, 9, 10, 11, 12};
    int h_col_idx[E] = {1, 2, 0, 3, 4, 0, 5, 6, 1, 1}; 
    int h_visited[N] = {0};
    int h_frontier[N] = {1, 0, 0, 0, 0, 0, 0}; // start from node 0
    int h_next_frontier[N] = {0};
    int h_done;

    int *d_row_ptr, *d_col_idx, *d_visited, *d_frontier, *d_next_frontier, *d_done;
    cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, E * sizeof(int));
    cudaMalloc(&d_visited, N * sizeof(int));
    cudaMalloc(&d_frontier, N * sizeof(int));
    cudaMalloc(&d_next_frontier, N * sizeof(int));
    cudaMalloc(&d_done, sizeof(int));

    cudaMemcpy(d_row_ptr, h_row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier, h_frontier, N * sizeof(int), cudaMemcpyHostToDevice);

    int level = 0;
    do {
        h_done = 1;
        cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice);

        bfs_kernel<<<1, N>>>(d_row_ptr, d_col_idx, d_visited, d_frontier, d_next_frontier, d_done);
        cudaMemcpy(d_frontier, d_next_frontier, N * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemset(d_next_frontier, 0, N * sizeof(int));
        cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);

        level++;
    } while (!h_done);

    cudaMemcpy(h_visited, d_visited, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Visited nodes (BFS from node 0): ";
    for (int i = 0; i < N; i++) std::cout << h_visited[i] << " ";
    std::cout << "\n";

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_visited);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_done);
    return 0;
}
