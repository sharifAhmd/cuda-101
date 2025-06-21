#include <cuda_runtime.h>
#include <stdio.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = (x > 0.0f) ? x : alpha * x;
    }
}

void print_array(const char* label, const float* arr, int n) {
    printf("%s: [", label);
    for (int i = 0; i < n; ++i)
        printf(" %.6f", arr[i]);
    printf(" ]\n");
}

int main() {
    const int N = 16;
    float alpha = 0.01f;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; ++i)
        h_in[i] = i - 8;

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128, blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(d_in, d_out, N, alpha);
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    print_array("Input", h_in, N);
    print_array("LeakyReLU", h_out, N);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
