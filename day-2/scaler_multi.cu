#include <iostream>
#define N 10

__global__ void scalarMultiply(int *input, int *output, int scalar) {
    // Calculate the global thread ID
    // Each thread will process one element of the array
    // The global thread ID is calculated using the block and thread indices
    // threadIdx.x is the thread index within the block
    // blockIdx.x is the block index
    // blockDim.x is the number of threads in each block
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    output[idx] = input[idx] * scalar;
}


int main() {
    int a[N], b[N];
    int *d_a, *d_b;
    int scalar = 5;

    for (int i = 0; i < N; i++) {
        a[i] = i + 1;  
    }

    // Allocate memory on GPU
    // Each block has 5 threads, and there are 2 blocks
    // So, total of 10 threads will be launched
    // Each thread will process one element of the array

    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMalloc((void **)&d_b, N * sizeof(int));

    // Copy input array to GPU
    // The cudaMemcpy function is used to copy data between the host (CPU) and device (GPU)
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    // The <<<2, 5>>> syntax specifies the number of blocks and threads per block
    // In this case, we are launching 2 blocks with 5 threads each
    scalarMultiply<<<2, 5>>>(d_a, d_b, scalar); 

    // Copy output array back to CPU
    cudaMemcpy(b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost);


    for (int i = 0; i < N; i++) {
        std::cout << a[i] << " * " << scalar << " = " << b[i] << std::endl;
    }

    // Free GPU memory
    // The cudaFree function is used to free the memory allocated on the GPU
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
