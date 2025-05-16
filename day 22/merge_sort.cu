#include <iostream>
#define N 8

__device__ int binarySearch(int *arr, int len, int val) {
    int low = 0, high = len;
    while (low < high) {
        int mid = (low + high) / 2;
        if (val <= arr[mid])
            high = mid;
        else
            low = mid + 1;
    }
    return low;
}

__global__ void parallelMerge(int *A, int *B, int *C, int lenA, int lenB) {
    int tid = threadIdx.x;
    if (tid < lenA) {
        int pos = binarySearch(B, lenB, A[tid]);
        C[tid + pos] = A[tid];
    }
    if (tid < lenB) {
        int pos = binarySearch(A, lenA, B[tid]);
        C[tid + pos] = B[tid];
    }
}

int main() {
    int h_A[N / 2] = {1, 3, 5, 7};
    int h_B[N / 2] = {2, 4, 6, 8};
    int h_C[N] = {0};

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, (N / 2) * sizeof(int));
    cudaMalloc(&d_B, (N / 2) * sizeof(int));
    cudaMalloc(&d_C, N * sizeof(int));

    cudaMemcpy(d_A, h_A, (N / 2) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, (N / 2) * sizeof(int), cudaMemcpyHostToDevice);

    parallelMerge<<<1, N / 2>>>(d_A, d_B, d_C, N / 2, N / 2);

    cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Merged output:\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
