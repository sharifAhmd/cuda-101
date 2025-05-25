#include <iostream>

#define N 1024
#define TPB 256


template <typename T>
struct Square {
    __device__ T operator()(T x) const {
        return x * x;
    }
};

template <typename T>
struct MultiplyBy {
    T factor;
    __device__ MultiplyBy(T f) : factor(f) {}
    __device__ T operator()(T x) const {
        return x * factor;
    }
};


template <typename T, typename Op>
__global__ void mapKernel(const T* input, T* output, int N, Op op) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        output[idx] = op(input[idx]);
    }
}


int main() {
    float *h_input = new float[N];
    float *h_output = new float[N];

    for (int i = 0; i < N; i++) {
        h_input[i] = i * 1.0f;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N + TPB - 1) / TPB;

    Square<float> squareOp;
    mapKernel<<<blocks, TPB>>>(d_input, d_output, N, squareOp);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "First 5 squares:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << h_input[i] << "^2 = " << h_output[i] << "\n";
    }

    MultiplyBy<float> mult10Op(10.0f);
    mapKernel<<<blocks, TPB>>>(d_input, d_output, N, mult10Op);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\nFirst 5 multiplied by 10:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << h_input[i] << " * 10 = " << h_output[i] << "\n";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return 0;
}
