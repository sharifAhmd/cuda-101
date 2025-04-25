#include <stdio.h>

__global__ void gpu_welcome_message(){
    // Calculate the global thread ID
    int thread_global_id = threadIdx.x + blockIdx.x * blockDim.x;

    printf("GPU says hello from Block %d, Thread %d (Global Thread ID: %d)!\n",
           blockIdx.x, threadIdx.x, thread_global_id);
}

int main(){
    // Set the number of blocks and threads per block
    int num_blocks = 3;
    int threads_per_block = 5;

    printf("Launching CUDA Kernel with %d blocks and %d threads per block.\n\n",
           num_blocks, threads_per_block);
    // Launch the kernel
    // Each block has 5 threads, and there are 3 blocks
    // So, total of 15 threads will be launched
    
    gpu_welcome_message<<<num_blocks, threads_per_block>>>();
    // Wait for the GPU to finish before accessing the results
    // This is important to ensure that all threads have completed before the program exits
    // Otherwise, the program may terminate before the GPU has finished executing
    cudaDeviceSynchronize();

    printf("\nExecution completed.\n");

    return 0;
}
