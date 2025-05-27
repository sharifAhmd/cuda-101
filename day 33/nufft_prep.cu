#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define N_SAMPLES 256
#define GRID_SIZE 64
#define KERNEL_RADIUS 2.5f

/*nvcc nufft_prep.cu -o nufft_prep -lm
Have to use -lm if using math constants like M_PI
*/

__device__ float kaiser_window(float dx, float dy, float beta = 13.5f) {
    float r2 = dx * dx + dy * dy;
    if (r2 > KERNEL_RADIUS * KERNEL_RADIUS) return 0.0f;

    float r = sqrtf(r2) / KERNEL_RADIUS;
    float denom = cyl_bessel_i0f(beta);
    return cyl_bessel_i0f(beta * sqrtf(1 - r * r)) / denom;
}


__global__ void nufft_grid(
    const float2* kspace_coords,
    const float2* samples,
    const float* density_weights,
    float2* grid,
    int grid_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_SAMPLES) return;

    float2 coord = kspace_coords[i];
    float2 val = samples[i];
    float weight = density_weights[i];

    float gx = (coord.x + 1.0f) * 0.5f * grid_size;
    float gy = (coord.y + 1.0f) * 0.5f * grid_size;

    int x0 = max(0, int(gx - KERNEL_RADIUS));
    int x1 = min(grid_size - 1, int(gx + KERNEL_RADIUS));
    int y0 = max(0, int(gy - KERNEL_RADIUS));
    int y1 = min(grid_size - 1, int(gy + KERNEL_RADIUS));

    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            float dx = gx - x;
            float dy = gy - y;
            float w = kaiser_window(dx, dy) * weight;

            int idx = y * grid_size + x;
            atomicAdd(&grid[idx].x, w * val.x);
            atomicAdd(&grid[idx].y, w * val.y);
        }
    }
}

int main() {
    float2 h_coords[N_SAMPLES];
    float2 h_samples[N_SAMPLES];
    float h_density[N_SAMPLES];

    for (int i = 0; i < N_SAMPLES; i++) {
        float t = i * 2 * M_PI / N_SAMPLES;
        float r = float(i) / N_SAMPLES;
        h_coords[i] = {r * cosf(t), r * sinf(t)};
        h_samples[i] = {cosf(t), sinf(t)};
        h_density[i] = 1.0f + 0.5f * r;  
    }

    float2 *d_coords, *d_samples, *d_grid;
    float *d_density;
    cudaMalloc(&d_coords, N_SAMPLES * sizeof(float2));
    cudaMalloc(&d_samples, N_SAMPLES * sizeof(float2));
    cudaMalloc(&d_density, N_SAMPLES * sizeof(float));
    cudaMalloc(&d_grid, GRID_SIZE * GRID_SIZE * sizeof(float2));
    cudaMemcpy(d_coords, h_coords, N_SAMPLES * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_samples, h_samples, N_SAMPLES * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_density, h_density, N_SAMPLES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_grid, 0, GRID_SIZE * GRID_SIZE * sizeof(float2));

    int threads = 128;
    int blocks = (N_SAMPLES + threads - 1) / threads;
    nufft_grid<<<blocks, threads>>>(d_coords, d_samples, d_density, d_grid, GRID_SIZE);
    cudaDeviceSynchronize();

    float2 h_result[GRID_SIZE * GRID_SIZE];
    cudaMemcpy(h_result, d_grid, GRID_SIZE * GRID_SIZE * sizeof(float2), cudaMemcpyDeviceToHost);

    std::cout << "Central grid value (approx image center): ";
    int center = (GRID_SIZE / 2) * GRID_SIZE + (GRID_SIZE / 2);
    std::cout << "(" << h_result[center].x << ", " << h_result[center].y << ")\n";

    cudaFree(d_coords);
    cudaFree(d_samples);
    cudaFree(d_density);
    cudaFree(d_grid);
    return 0;
}
