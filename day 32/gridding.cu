#include <iostream>
#include <cuda_runtime.h>

#define N_SAMPLES 8
#define GRID_SIZE 8
#define KERNEL_RADIUS 1.5f


__device__ float kernel(float dx, float dy) {
    float r2 = dx * dx + dy * dy;
    return (r2 <= KERNEL_RADIUS * KERNEL_RADIUS) ? 1.0f : 0.0f;
}


__global__ void gridSamples(
    const float2* kspace_coords, const float* samples,
    float* grid_real, float* grid_imag,
    int grid_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N_SAMPLES) return;

    float2 coord = kspace_coords[i];  
    float real = samples[i];
    float imag = 0.0f;  

    float gx = (coord.x + 1.0f) * grid_size / 2.0f;
    float gy = (coord.y + 1.0f) * grid_size / 2.0f;

    int x0 = max(0, int(gx - KERNEL_RADIUS));
    int x1 = min(grid_size - 1, int(gx + KERNEL_RADIUS));
    int y0 = max(0, int(gy - KERNEL_RADIUS));
    int y1 = min(grid_size - 1, int(gy + KERNEL_RADIUS));

    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            float dx = gx - x;
            float dy = gy - y;
            float w = kernel(dx, dy);

            int idx = y * grid_size + x;
            atomicAdd(&grid_real[idx], w * real);
            atomicAdd(&grid_imag[idx], w * imag);
        }
    }
}

int main() {
    float2 h_coords[N_SAMPLES] = {
        {-0.5f, -0.5f}, {-0.2f, -0.1f}, {0.1f, 0.1f}, {0.3f, 0.4f},
        {-0.3f, 0.3f}, {0.0f, 0.0f}, {0.5f, 0.5f}, {0.7f, -0.7f}
    };
    float h_samples[N_SAMPLES] = {1, 2, 3, 4, 2, 1, 0.5, 0.2};

    float *d_grid_real, *d_grid_imag;
    float2 *d_coords;
    float *d_samples;

    cudaMalloc(&d_coords, N_SAMPLES * sizeof(float2));
    cudaMalloc(&d_samples, N_SAMPLES * sizeof(float));
    cudaMalloc(&d_grid_real, GRID_SIZE * GRID_SIZE * sizeof(float));
    cudaMalloc(&d_grid_imag, GRID_SIZE * GRID_SIZE * sizeof(float));

    cudaMemcpy(d_coords, h_coords, N_SAMPLES * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_samples, h_samples, N_SAMPLES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_grid_real, 0, GRID_SIZE * GRID_SIZE * sizeof(float));
    cudaMemset(d_grid_imag, 0, GRID_SIZE * GRID_SIZE * sizeof(float));

    gridSamples<<<1, N_SAMPLES>>>(d_coords, d_samples, d_grid_real, d_grid_imag, GRID_SIZE);
    cudaDeviceSynchronize();

    float h_result[GRID_SIZE * GRID_SIZE];
    cudaMemcpy(h_result, d_grid_real, GRID_SIZE * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Gridded real values:\n";
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            std::cout << h_result[y * GRID_SIZE + x] << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_coords);
    cudaFree(d_samples);
    cudaFree(d_grid_real);
    cudaFree(d_grid_imag);
    return 0;
}
