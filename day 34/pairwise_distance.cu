#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define N 1024  
struct Atom {
    float x, y, z;
};


__global__ void computeDistances(const Atom* atoms, float* distances, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        float dx = atoms[i].x - atoms[j].x;
        float dy = atoms[i].y - atoms[j].y;
        float dz = atoms[i].z - atoms[j].z;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);

        distances[i * N + j] = dist;
    }
}


int main() {
    Atom* h_atoms = new Atom[N];
    for (int i = 0; i < N; i++) {
        h_atoms[i].x = rand() / (float)RAND_MAX;
        h_atoms[i].y = rand() / (float)RAND_MAX;
        h_atoms[i].z = rand() / (float)RAND_MAX;
    }

    Atom* d_atoms;
    float* d_distances;
    cudaMalloc(&d_atoms, N * sizeof(Atom));
    cudaMalloc(&d_distances, N * N * sizeof(float));

    cudaMemcpy(d_atoms, h_atoms, N * sizeof(Atom), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);


    computeDistances<<<numBlocks, threadsPerBlock>>>(d_atoms, d_distances, N);
    cudaDeviceSynchronize();

    float result = 0;
    cudaMemcpy(&result, d_distances + 0 * N + 1, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Distance between atom 0 and atom 1: " << result << "\n";

    cudaFree(d_atoms);
    cudaFree(d_distances);
    delete[] h_atoms;

    return 0;
}
