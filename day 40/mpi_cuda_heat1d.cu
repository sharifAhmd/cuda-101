#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 1024           
#define STEPS 100        
#define TPB 256          

__global__ void heat_update(float* curr, float* next, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (idx < n - 1) {
        next[idx] = 0.5f * curr[idx] + 0.25f * (curr[idx - 1] + curr[idx + 1]);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cudaSetDevice(rank); 

    int local_N = N / size + 2;  
    float *h_data = new float[local_N];


    for (int i = 1; i < local_N - 1; i++) {
        h_data[i] = rank;
    }
    h_data[0] = h_data[local_N - 1] = 0.0f;

    float *d_curr, *d_next;
    cudaMalloc(&d_curr, local_N * sizeof(float));
    cudaMalloc(&d_next, local_N * sizeof(float));
    cudaMemcpy(d_curr, h_data, local_N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (local_N - 2 + TPB - 1) / TPB;

    for (int step = 0; step < STEPS; step++) {

        cudaMemcpy(h_data, d_curr, local_N * sizeof(float), cudaMemcpyDeviceToHost);


        if (rank > 0) {
            MPI_Send(&h_data[1], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&h_data[0], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Send(&h_data[local_N - 2], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&h_data[local_N - 1], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        cudaMemcpy(d_curr, h_data, local_N * sizeof(float), cudaMemcpyHostToDevice);
        heat_update<<<blocks, TPB>>>(d_curr, d_next, local_N);
        std::swap(d_curr, d_next);
    }

    cudaMemcpy(h_data, d_curr, local_N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Rank " << rank << " center value: " << h_data[local_N / 2] << std::endl;

    cudaFree(d_curr);
    cudaFree(d_next);
    delete[] h_data;

    MPI_Finalize();
    return 0;
}
