#include <iostream>
#define N 8
#define THREADS 4

__device__ int mergePathSearch(int diagonal, int *A, int lenA, int *B, int lenB) {
    int low = max(0, diagonal - lenB);
    int high = min(diagonal, lenA);

    while (low < high) {
        int mid = (low + high) / 2;
        if (A[mid] < B[diagonal - mid - 1])
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

__global__ void mergePathKernel(int *A, int *B, int *C, int lenA, int lenB) {
    int tid = threadIdx.x;
    int total = lenA + lenB;
    int elementsPerThread = (total + blockDim.x - 1) / blockDim.x;

    int diag_start = tid * elementsPerThread;
    int diag_end = min(diag_start + elementsPerThread, total);

    int a_start = mergePathSearch(diag_start, A, lenA, B, lenB);
    int b_start = diag_start - a_start;

    int a_end = mergePathSearch(diag_end, A, lenA, B, lenB);
    int b_end = diag_end - a_end;

    int i = a_start, j = b_start, k = diag_start;
    while (i < a_end && j < b_end) {
        C[k++] = (A[i] < B[j]) ? A[i++] : B[j++];
    }
    while (i < a_end) C[k++] = A[i++];
    while (j < b_end) C[k++] = B[j++];
}

int main() {
    int h_A[N / 2] = {1, 3, 5, 7};
    int h_B[N / 2] = {2, 4, 6, 8};
    int h_C[N];

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, (N / 2) * sizeof(int));
    cudaMalloc(&d_B, (N / 2) * sizeof(int));
    cudaMalloc(&d_C, N * sizeof(int));

    cudaMemcpy(d_A, h_A, (N / 2) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, (N / 2) * sizeof(int), cudaMemcpyHostToDevice);

    mergePathKernel<<<1, THREADS>>>(d_A, d_B, d_C, N / 2, N / 2);

    cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Merged output (via merge path): ";
    for (int i = 0; i < N; i++) std::cout << h_C[i] << " ";
    std::cout << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
