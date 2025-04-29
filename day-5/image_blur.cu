#include <iostream>
#include <cstdlib>

#define WIDTH 8
#define HEIGHT 8
#define CHANNELS 3

__global__ void blurRGB(unsigned char *input, unsigned char *output, int width, int height) {
    
    /**
     * Performs image blurring using a 3x3 averaging filter
     * For each pixel, calculates the average value of its 3x3 neighborhood
     * The kernel processes each color channel separately
     * Edge pixels use fewer samples in the average calculation
     */

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * CHANNELS;

    for (int c = 0; c < CHANNELS; c++) {
        int sum = 0;
        int count = 0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int n_idx = (ny * width + nx) * CHANNELS + c;
                    sum += input[n_idx];
                    count++;
                }
            }
        }

        output[idx + c] = sum / count;
    }
}

int main() {
    const int imageSize = WIDTH * HEIGHT * CHANNELS;

    unsigned char h_input[imageSize];
    unsigned char h_output[imageSize];

    for (int i = 0; i < imageSize; i++) {
        h_input[i] = rand() % 256;
    }

    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, imageSize);
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks((WIDTH + 7) / 8, (HEIGHT + 7) / 8);
    blurRGB<<<numBlocks, threadsPerBlock>>>(d_input, d_output, WIDTH, HEIGHT);

    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    std::cout << "Original RGB\t\tBlurred RGB" << std::endl;
    for (int i = 0; i < 8; i++) {
        int base = i * CHANNELS;
        std::cout << "(" << (int)h_input[base] << ", " << (int)h_input[base + 1] << ", " << (int)h_input[base + 2] << ")\t\t";
        std::cout << "(" << (int)h_output[base] << ", " << (int)h_output[base + 1] << ", " << (int)h_output[base + 2] << ")" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
