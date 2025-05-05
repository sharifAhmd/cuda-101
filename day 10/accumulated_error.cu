#include <iostream>

#define N 1000000

// Two different approaches to summing an array of floating-point numbers
// 1. A naive summation method which accumulate significant floating-point rounding errors.
// 2. The Kahan summation algorithm, which reduces rounding errors by compensating for lost low-order bits.
// Both kernels operate on a large array of small floating-point values to highlight the difference in numerical accuracy.
__global__ void naiveSum(float *arr, float *result) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += arr[i];
    }
    result[0] = sum;
}

__global__ void kahanSum(float *arr, float *result) {
    float sum = 0.0f;
    float c = 0.0f;
    for (int i = 0; i < N; i++) {
        float y = arr[i] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    result[0] = sum;
}

int main() {

    float *h_arr = new float[N];
    for (int i = 0; i < N; i++) h_arr[i] = 0.0001f;

    float naiveResult, kahanResult;
    float *d_arr, *d_result;

    cudaMalloc(&d_arr, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice);

    naiveSum<<<1,1>>>(d_arr, d_result);
    cudaMemcpy(&naiveResult, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    kahanSum<<<1,1>>>(d_arr, d_result);
    cudaMemcpy(&kahanResult, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Naive Sum: " << naiveResult << std::endl;
    std::cout << "Kahan Sum: " << kahanResult << std::endl;

    cudaFree(d_arr);
    cudaFree(d_result);
    delete[] h_arr;
    return 0;
}
