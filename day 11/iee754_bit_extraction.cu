#include <iostream>
#include <iomanip>  

#define N 4

/**
 * The kernel processes an array of floating-point numbers 
 * in parallel, extracting the sign, exponent, and mantissa for each number.
 *The kernel uses the `__float_as_int` intrinsic to reinterpret the bits of a float as an integer.
 *The sign bit is extracted by shifting and masking, the exponent is extracted by shifting and masking,
 *and the mantissa is extracted by masking the lower 23 bits.
 */
__global__ void extractIEEE754(float *input, int *sign, int *exponent, int *mantissa) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int bits = __float_as_int(input[idx]);  

    sign[idx] = (bits >> 31) & 0x1;
    exponent[idx] = (bits >> 23) & 0xFF;
    mantissa[idx] = bits & 0x7FFFFF;
}

int main() {
    float h_input[N] = {1.0f, -2.5f, 0.15625f, 123.456f};
    int h_sign[N], h_exponent[N], h_mantissa[N];

    float *d_input;
    int *d_sign, *d_exponent, *d_mantissa;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_sign, N * sizeof(int));
    cudaMalloc(&d_exponent, N * sizeof(int));
    cudaMalloc(&d_mantissa, N * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    extractIEEE754<<<1, N>>>(d_input, d_sign, d_exponent, d_mantissa);

    cudaMemcpy(h_sign, d_sign, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_exponent, d_exponent, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mantissa, d_mantissa, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << "Input: " << h_input[i] << "\n";
        std::cout << "  Sign: " << h_sign[i] << "\n";
        std::cout << "  Exponent (decimal): " << h_exponent[i] << " (raw bias)\n";
        std::cout << "  Mantissa (hex): 0x" << std::hex << h_mantissa[i] << std::dec << "\n\n";
    }

    cudaFree(d_input);
    cudaFree(d_sign);
    cudaFree(d_exponent);
    cudaFree(d_mantissa);

    return 0;
}
