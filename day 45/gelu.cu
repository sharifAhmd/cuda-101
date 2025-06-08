#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float c = 0.044715f;
        float sqrt_2_over_pi = 0.7978845608f;
        float x3 = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + c * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(tanh_arg));
    }
}

void print_array(const char* name, const float* arr, int n) {
    printf("%s: [", name);
    for (int i = 0; i < n; ++i) printf(" %.4f", arr[i]);
    printf(" ]\n");
}

int main() {

    const int N = 8;
    float h_in[N]  = { -2, -1, 0, 1, 2, 3, 4, 5 };
    float h_out[N];

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);


    gelu_kernel<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    print_array("Input", h_in, N);
    print_array("GELU", h_out, N);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
