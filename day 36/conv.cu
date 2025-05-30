#include <iostream>
#include <cuda_runtime.h>

#define IMG_WIDTH 8
#define IMG_HEIGHT 8
#define KERNEL_SIZE 3
#define TILE_WIDTH 6 

__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ void conv2d_shared(const float* input, float* output, int width, int height) {
    __shared__ float tile[IMG_HEIGHT][IMG_WIDTH];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        tile[y][x] = input[y * width + x];
    }

    __syncthreads();

    if (x < width - 2 && y < height - 2) {
        float sum = 0.0f;
        for (int ky = 0; ky < KERNEL_SIZE; ky++) {
            for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                float pixel = tile[y + ky][x + kx];
                float weight = d_kernel[ky * KERNEL_SIZE + kx];
                sum += pixel * weight;
            }
        }
        output[y * width + x] = sum;
    }
}

int main() {
    float h_image[IMG_HEIGHT * IMG_WIDTH] = {
        1, 1, 2, 4, 8, 8, 8, 8,
        1, 1, 2, 4, 8, 8, 8, 8,
        1, 1, 2, 4, 8, 8, 8, 8,
        1, 1, 2, 4, 8, 8, 8, 8,
        1, 1, 2, 4, 8, 8, 8, 8,
        1, 1, 2, 4, 8, 8, 8, 8,
        1, 1, 2, 4, 8, 8, 8, 8,
        1, 1, 2, 4, 8, 8, 8, 8,
    };

    float h_kernel[KERNEL_SIZE * KERNEL_SIZE] = {
         1,  0, -1,
         1,  0, -1,
         1,  0, -1
    }; 

    float h_output[IMG_HEIGHT * IMG_WIDTH] = {0};

    float *d_image, *d_output;
    cudaMalloc(&d_image, IMG_HEIGHT * IMG_WIDTH * sizeof(float));
    cudaMalloc(&d_output, IMG_HEIGHT * IMG_WIDTH * sizeof(float));
    cudaMemcpy(d_image, h_image, IMG_HEIGHT * IMG_WIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    dim3 threads(8, 8);
    dim3 blocks((IMG_WIDTH + threads.x - 1) / threads.x, (IMG_HEIGHT + threads.y - 1) / threads.y);

    conv2d_shared<<<blocks, threads>>>(d_image, d_output, IMG_WIDTH, IMG_HEIGHT);
    cudaMemcpy(h_output, d_output, IMG_HEIGHT * IMG_WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Output Feature Map:\n";
    for (int y = 0; y < IMG_HEIGHT; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            std::cout << h_output[y * IMG_WIDTH + x] << "\t";
        }
        std::cout << "\n";
    }

    cudaFree(d_image);
    cudaFree(d_output);
    return 0;
}
