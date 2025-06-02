#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 1024  
#define TPB 256

__global__ void partialSum(const float* input, float* output, int n) {
    __shared__ float buffer[TPB];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n)
        buffer[tid] = input[idx];
    else
        buffer[tid] = 0.0f;

    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            buffer[tid] += buffer[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = buffer[0];
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cudaSetDevice(rank);  

    int local_N = N / size;

    float* h_data = new float[local_N];
    for (int i = 0; i < local_N; i++) {
        h_data[i] = 1.0f; 
    }

    float *d_data, *d_partial, *h_partial;
    cudaMalloc(&d_data, local_N * sizeof(float));
    cudaMemcpy(d_data, h_data, local_N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (local_N + TPB - 1) / TPB;
    cudaMalloc(&d_partial, blocks * sizeof(float));
    h_partial = new float[blocks];

    partialSum<<<blocks, TPB>>>(d_data, d_partial, local_N);
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float local_sum = 0.0f;
    for (int i = 0; i < blocks; i++) {
        local_sum += h_partial[i];
    }

    float global_sum = 0.0f;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Global sum: " << global_sum << std::endl;
    }

    cudaFree(d_data);
    cudaFree(d_partial);
    delete[] h_data;
    delete[] h_partial;

    MPI_Finalize();
    return 0;
}
