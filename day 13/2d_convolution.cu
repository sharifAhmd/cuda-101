#include <iostream>
#define WIDTH 8
#define HEIGHT 8

/**
 * Performs a 2D convolution operation on the input array.
 */
__global__ void stencil2D(int *input, int *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;
        output[idx] = input[idx] + 
                      input[(y - 1) * width + x] + // top
                      input[(y + 1) * width + x] + // bottom
                      input[y * width + (x - 1)] + // left
                      input[y * width + (x + 1)];  // right
    }
}

int main() {
    int size = WIDTH * HEIGHT;
    int h_input[size], h_output[size];

    for (int i = 0; i < size; i++) {
        h_input[i] = 1;  
    }

    int *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    stencil2D<<<numBlocks, threadsPerBlock>>>(d_input, d_output, WIDTH, HEIGHT);

    cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Stencil output:\n";
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            std::cout << h_output[y * WIDTH + x] << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
