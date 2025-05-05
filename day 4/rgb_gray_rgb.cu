#include <iostream>
#include <cstdlib> // For rand()

#define WIDTH 4
#define HEIGHT 4
#define N (WIDTH * HEIGHT)


__global__ void rgb_to_gray(unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *gray, int numPixels) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPixels) {
        gray[idx] = 0.299f * r[idx] + 0.587f * g[idx] + 0.114f * b[idx];
    }
}


__global__ void gray_to_rgb(unsigned char *gray, unsigned char *r, unsigned char *g, unsigned char *b, int numPixels) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPixels) {
        r[idx] = gray[idx];
        g[idx] = gray[idx];
        b[idx] = gray[idx];
    }
}

int main() {

    unsigned char h_r[N], h_g[N], h_b[N], h_gray[N];
    unsigned char h_r_out[N], h_g_out[N], h_b_out[N];

    // Initialize random pixel values for RGB channels
    for (int i = 0; i < N; i++) {
        h_r[i] = rand() % 256;  
        h_g[i] = rand() % 256;
        h_b[i] = rand() % 256;
    }

    // Device arrays
    unsigned char *d_r, *d_g, *d_b, *d_gray;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_r, N * sizeof(unsigned char));
    cudaMalloc((void**)&d_g, N * sizeof(unsigned char));
    cudaMalloc((void**)&d_b, N * sizeof(unsigned char));
    cudaMalloc((void**)&d_gray, N * sizeof(unsigned char));

    // Copy original RGB arrays to device
    cudaMemcpy(d_r, h_r, N * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g, N * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Launch rgb_to_gray kernel
    int blockSize = 16;
    int numBlocks = (N + blockSize - 1) / blockSize;
    rgb_to_gray<<<numBlocks, blockSize>>>(d_r, d_g, d_b, d_gray, N);

    // Copy gray result back to host
    cudaMemcpy(h_gray, d_gray, N * sizeof(unsigned char), cudaMemcpyDeviceToHost);


    gray_to_rgb<<<numBlocks, blockSize>>>(d_gray, d_r, d_g, d_b, N);

    // Copy the new RGB output arrays back to host
    cudaMemcpy(h_r_out, d_r, N * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_out, d_g, N * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_out, d_b, N * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Print the results
    std::cout << "Original RGB\tGray\tRestored RGB" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << "(" << (int)h_r[i] << "," << (int)h_g[i] << "," << (int)h_b[i] << ")\t"
                  << (int)h_gray[i] << "\t"
                  << "(" << (int)h_r_out[i] << "," << (int)h_g_out[i] << "," << (int)h_b_out[i] << ")"
                  << std::endl;
    }

    // Free device memory
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_gray);

    return 0;
}
