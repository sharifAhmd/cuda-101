#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define N 512               
#define MAX_NEIGHBORS 64    
#define CUTOFF 0.2f         

struct Atom {
    float x, y, z;
};

__global__ void findNeighbors(const Atom* atoms, int* neighbors, int* neighbor_counts, int N, float cutoff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    int count = 0;
    for (int j = 0; j < N && count < MAX_NEIGHBORS; j++) {
        if (i == j) continue;
        
        float dx = atoms[i].x - atoms[j].x;
        float dy = atoms[i].y - atoms[j].y;
        float dz = atoms[i].z - atoms[j].z;
        float dist2 = dx * dx + dy * dy + dz * dz;
        
        if (dist2 <= cutoff * cutoff) {
            neighbors[i * MAX_NEIGHBORS + count] = j;
            count++;
        }
    }
    neighbor_counts[i] = count;
}

int main() {
    Atom* h_atoms = new Atom[N];
    for (int i = 0; i < N; i++) {
        h_atoms[i].x = rand() / (float)RAND_MAX;
        h_atoms[i].y = rand() / (float)RAND_MAX;
        h_atoms[i].z = rand() / (float)RAND_MAX;
    }
    
    Atom* d_atoms;
    int *d_neighbors, *d_neighbor_counts;
    cudaMalloc(&d_atoms, N * sizeof(Atom));
    cudaMalloc(&d_neighbors, N * MAX_NEIGHBORS * sizeof(int));
    cudaMalloc(&d_neighbor_counts, N * sizeof(int));
    
    cudaMemcpy(d_atoms, h_atoms, N * sizeof(Atom), cudaMemcpyHostToDevice);
    cudaMemset(d_neighbors, -1, N * MAX_NEIGHBORS * sizeof(int));
    cudaMemset(d_neighbor_counts, 0, N * sizeof(int));
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    findNeighbors<<<blocks, threads>>>(d_atoms, d_neighbors, d_neighbor_counts, N, CUTOFF);
    
    cudaDeviceSynchronize();
    
    int h_counts[N];
    int h_neighbors[N * MAX_NEIGHBORS];
    cudaMemcpy(h_counts, d_neighbor_counts, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_neighbors, d_neighbors, N * MAX_NEIGHBORS * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "Atom 0 has " << h_counts[0] << " neighbors within cutoff:\n";
    for (int i = 0; i < h_counts[0]; i++) {
        std::cout << "  -> Atom " << h_neighbors[i] << "\n";
    }
    
    cudaFree(d_atoms);
    cudaFree(d_neighbors);
    cudaFree(d_neighbor_counts);
    delete[] h_atoms;
    
    return 0;
}