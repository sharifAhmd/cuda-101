#include <iostream>
#include <climits>
#define N 5
#define E 8
#define INF 1e9

__global__ void relax_edges(int *src, int *dst, int *weights, int *dist, bool *updated) {
    int tid = threadIdx.x;
    int u = src[tid];
    int v = dst[tid];
    int w = weights[tid];

    if (dist[u] != INF && dist[u] + w < dist[v]) {
        dist[v] = dist[u] + w;
        *updated = true;
    }
}

int main() {
    int h_src[E]    = {0, 0, 1, 1, 1, 2, 3, 4};
    int h_dst[E]    = {1, 2, 2, 3, 4, 4, 4, 3};
    int h_weights[E]= {6, 7, 5, -4, 8, -3, 9, 7};

    int h_dist[N];
    for (int i = 0; i < N; i++) h_dist[i] = INF;
    h_dist[0] = 0; 

    int *d_src, *d_dst, *d_weights, *d_dist;
    bool h_updated, *d_updated;

    cudaMalloc(&d_src, E * sizeof(int));
    cudaMalloc(&d_dst, E * sizeof(int));
    cudaMalloc(&d_weights, E * sizeof(int));
    cudaMalloc(&d_dist, N * sizeof(int));
    cudaMalloc(&d_updated, sizeof(bool));

    cudaMemcpy(d_src, h_src, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, h_dist, N * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < N - 1; i++) {
        h_updated = false;
        cudaMemcpy(d_updated, &h_updated, sizeof(bool), cudaMemcpyHostToDevice);

        relax_edges<<<1, E>>>(d_src, d_dst, d_weights, d_dist, d_updated);

        cudaMemcpy(&h_updated, d_updated, sizeof(bool), cudaMemcpyDeviceToHost);
        if (!h_updated) break; // Early exit if no updates
    }

    cudaMemcpy(h_dist, d_dist, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Shortest distances from node 0:\n";
    for (int i = 0; i < N; i++) {
        std::cout << "Node " << i << ": ";
        if (h_dist[i] == INF) std::cout << "INF\n";
        else std::cout << h_dist[i] << "\n";
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_weights);
    cudaFree(d_dist);
    cudaFree(d_updated);
    return 0;
}
