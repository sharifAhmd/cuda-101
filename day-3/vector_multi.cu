#include <iostream>

#define N 8 


__global__ void vectorMultiply(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        c[idx] = a[idx] * b[idx];
    }
}


int main() {
    int a[N], b[N], c[N];        
    int *d_a, *d_b, *d_c;          


    for (int i = 0; i < N; i++) {
        a[i] = i + 1;              
        b[i] = (i + 1) * 2;        
    }


    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));


    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    vectorMultiply<<<2, 4>>>(d_a, d_b, d_c); 

    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Vector Multiplication Results:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << a[i] << " * " << b[i] << " = " << c[i] << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
