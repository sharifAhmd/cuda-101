#include <iostream>
#define N 16

__global__ void vectorAddShared(int *a, int *b, int *c) {
    /**
     * This kernel adds two vectors using shared memory
     * Each thread loads one element from global memory to shared memory
     * Then, it performs the addition using the values in shared memory
     * Finally, it writes the result back to global memory
     */
    __shared__ int tile_a[256];
    __shared__ int tile_b[256];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Load from global to shared memory
    tile_a[threadIdx.x] = a[idx];
    tile_b[threadIdx.x] = b[idx];
    __syncthreads();  // ensure all threads finish loading

    // Perform computation
    int temp = tile_a[threadIdx.x] + tile_b[threadIdx.x];

    // Write back to global memory
    c[idx] = temp;
}

int main() {
    int h_a[N], h_b[N], h_c[N];
    int *d_a, *d_b, *d_c;

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    vectorAddShared<<<1, N>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Result:\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << "\n";
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
