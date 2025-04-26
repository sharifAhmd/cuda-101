#include <iostream>
#define N 8

__global__ void squareElements(int *input, int *output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    output[idx] = input[idx] * input[idx];
}


int main() {
    int a[N], b[N];
    int *d_a, *d_b;


    for (int i = 0; i < N; i++) {
        a[i] = i + 1;   // {1, 2, 3, 4, 5, 6, 7, 8}
    }

    // Allocate memory on GPU
    // Each block has 4 threads, and there are 2 blocks
    // So, total of 8 threads will be launched
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

    squareElements<<<2, 4>>>(d_a, d_b);

    cudaMemcpy(b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Squares of array elements:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << a[i] << "^2 = " << b[i] << std::endl;
    }


    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
