#include <iostream>
#include <cuda_runtime.h>

#define N 16

__device__ void swap(int &a, int &b) {
    int t = a;
    a = b;
    b = t;
}

__global__ void quicksort(int *data, int left, int right) {
    if (left >= right) return;

    int pivot = data[(left + right) / 2];
    int i = left;
    int j = right;

    while (i <= j) {
        while (data[i] < pivot) i++;
        while (data[j] > pivot) j--;
        if (i <= j) {
            swap(data[i], data[j]);
            i++; 
            j--;
        }
    }

    if (left < j) {
        quicksort<<<1, 1>>>(data, left, j);
    }
    if (i < right) {
        quicksort<<<1, 1>>>(data, i, right);
    }
}

int main() {
    int h_data[N] = {23, 1, 45, 34, 7, 5, 19, 99, 13, 55, 2, 77, 84, 31, 8, 20};

    int *d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the initial quicksort
    quicksort<<<1, 1>>>(d_data, 0, N - 1);
    cudaDeviceSynchronize();  

    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sorted Output:\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << "\n";

    cudaFree(d_data);
    return 0;
}
