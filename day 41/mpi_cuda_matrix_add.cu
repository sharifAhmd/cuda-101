#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 1024       
#define TPB 256     

__global__ void matrix_add(const float* A, const float* B, float* C, int total_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_cols) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cudaSetDevice(rank);  

    int rows_per_proc = N / size;
    int chunk_size = rows_per_proc * N;

    float *h_A = new float[chunk_size];
    float *h_B = new float[chunk_size];
    float *h_C = new float[chunk_size];

    for (int i = 0; i < chunk_size; i++) {
        h_A[i] = 1.0f;
        h_B[i] = rank;  
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, chunk_size * sizeof(float));
    cudaMalloc(&d_B, chunk_size * sizeof(float));
    cudaMalloc(&d_C, chunk_size * sizeof(float));

    cudaMemcpy(d_A, h_A, chunk_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, chunk_size * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (chunk_size + TPB - 1) / TPB;
    matrix_add<<<blocks, TPB>>>(d_A, d_B, d_C, chunk_size);
    cudaMemcpy(h_C, d_C, chunk_size * sizeof(float), cudaMemcpyDeviceToHost);


    float* gathered = nullptr;
    if (rank == 0) {
        gathered = new float[N * N];
    }

    MPI_Gather(h_C, chunk_size, MPI_FLOAT, gathered, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "First few elements of result matrix:\n";
        for (int i = 0; i < 10; i++) {
            std::cout << gathered[i] << " ";
        }
        std::cout << "\n";
        delete[] gathered;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    MPI_Finalize();
    return 0;
}
