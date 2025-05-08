#include <iostream>
#define WIDTH 8
#define HEIGHT 8

/**
 * Performs a 2D Gaussian blur operation on the input array.
 * The kernel size is 3x3 with a Gaussian distribution.
 */
__global__ void gaussianBlurKernel(float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float result = 0.0f;

        result += input[(y - 1) * width + (x - 1)] * 1.0f;
        result += input[(y - 1) * width + x]       * 2.0f;
        result += input[(y - 1) * width + (x + 1)] * 1.0f;

        result += input[y * width + (x - 1)]       * 2.0f;
        result += input[y * width + x]             * 4.0f;
        result += input[y * width + (x + 1)]       * 2.0f;

        result += input[(y + 1) * width + (x - 1)] * 1.0f;
        result += input[(y + 1) * width + x]       * 2.0f;
        result += input[(y + 1) * width + (x + 1)] * 1.0f;

        output[y * width + x] = result / 16.0f;
    }
}

int main() {
    const int size = WIDTH * HEIGHT;
    float h_input[size], h_output[size];

    for (int i = 0; i < size; i++) {
        h_input[i] = (i % WIDTH) * 10.0f; 
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gaussianBlurKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, WIDTH, HEIGHT);

    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Blurred Output:\n";
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            std::cout << h_output[y * WIDTH + x] << "\t";
        }
        std::cout << "\n";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
