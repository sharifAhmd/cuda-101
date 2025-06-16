#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void mish_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float sp = (x > 20) ? x : log1pf(expf(x));
        output[idx] = x * tanhf(sp);
    }
}

void print_array(const char* label, float* arr, int n) {
    printf("%s: [", label);
    for (int i = 0; i < n; ++i)
        printf(" %.6f", arr[i]);
    printf(" ]\n");
}

int main() {
    const int N = 8;
    float h_in[N] = { -5, -1, 0, 1, 2, 5, 10, 20 };
    float h_out[N];

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256, blocks = (N + threads - 1) / threads;
    mish_kernel<<<blocks, threads>>>(d_in, d_out, N);
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    print_array("Input", h_in, N);
    print_array("Mish", h_out, N);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
