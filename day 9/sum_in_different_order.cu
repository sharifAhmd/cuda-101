#include <iostream>

#define N 1000000

__global__ void sumForward(float *arr, float *result) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += arr[i];
    }
    result[0] = sum;
}

__global__ void sumBackward(float *arr, float *result) {
    float sum = 0.0f;
    for (int i = N - 1; i >= 0; i--) {
        sum += arr[i];
    }
    result[0] = sum;
}

int main() {
    float h_arr[N];
    for (int i = 0; i < N; i++) h_arr[i] = 1.0f / (i + 1);

    float *d_arr, *d_result;
    float h_result_forward, h_result_backward;

    cudaMalloc(&d_arr, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice);

    sumForward<<<1,1>>>(d_arr, d_result);
    cudaMemcpy(&h_result_forward, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    sumBackward<<<1,1>>>(d_arr, d_result);
    cudaMemcpy(&h_result_backward, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Sum Forward = " << h_result_forward << std::endl;
    std::cout << "Sum Backward = " << h_result_backward << std::endl;

    cudaFree(d_arr);
    cudaFree(d_result);
    return 0;
}
