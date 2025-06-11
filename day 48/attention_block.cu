#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define SEQ_LEN 4      
#define D_MODEL 4      
#define HEAD_DIM 4     

__device__ float cuda_exp(float x) {
    return expf(x);
}

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    // C[M x K] = A[M x N] * B[N x K]
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        float val = 0;
        for (int i = 0; i < N; ++i)
            val += A[row * N + i] * B[i * K + col];
        C[row * K + col] = val;
    }
}

__global__ void scale(float* mat, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        mat[idx] /= scale;
}


__global__ void row_softmax(float* mat, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        float maxval = -1e20;
        for (int i = 0; i < cols; ++i)
            if (mat[row * cols + i] > maxval)
                maxval = mat[row * cols + i];
        float sum = 0;
        for (int i = 0; i < cols; ++i) {
            mat[row * cols + i] = expf(mat[row * cols + i] - maxval);
            sum += mat[row * cols + i];
        }
        for (int i = 0; i < cols; ++i)
            mat[row * cols + i] /= sum;
    }
}

void print_matrix(const char* label, float* data, int rows, int cols) {
    printf("%s:\n", label);
    for (int r = 0; r < rows; ++r) {
        printf("[");
        for (int c = 0; c < cols; ++c)
            printf(" %.4f", data[r * cols + c]);
        printf(" ]\n");
    }
}

int main() {
    float h_X[SEQ_LEN * D_MODEL] = {
        1, 2, 3, 4,
        2, 3, 4, 1,
        3, 4, 1, 2,
        4, 1, 2, 3
    };

    float h_Wq[D_MODEL * HEAD_DIM] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };
    float h_Wk[D_MODEL * HEAD_DIM] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };
    float h_Wv[D_MODEL * HEAD_DIM] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };

    float *d_X, *d_Wq, *d_Wk, *d_Wv, *d_Q, *d_K, *d_V, *d_scores, *d_softmax, *d_out;
    cudaMalloc(&d_X,     SEQ_LEN * D_MODEL * sizeof(float));
    cudaMalloc(&d_Wq,    D_MODEL * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_Wk,    D_MODEL * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_Wv,    D_MODEL * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_Q,     SEQ_LEN * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_K,     SEQ_LEN * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_V,     SEQ_LEN * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_scores, SEQ_LEN * SEQ_LEN * sizeof(float));
    cudaMalloc(&d_softmax, SEQ_LEN * SEQ_LEN * sizeof(float));
    cudaMalloc(&d_out,   SEQ_LEN * HEAD_DIM * sizeof(float));

    cudaMemcpy(d_X,  h_X,  SEQ_LEN * D_MODEL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wq, h_Wq, D_MODEL * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wk, h_Wk, D_MODEL * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wv, h_Wv, D_MODEL * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // Q = XWq, K = XWk, V = XWv
    dim3 grid1(1, SEQ_LEN);
    dim3 block1(HEAD_DIM, 1);
    matmul<<<grid1, block1>>>(d_X, d_Wq, d_Q, SEQ_LEN, D_MODEL, HEAD_DIM);
    matmul<<<grid1, block1>>>(d_X, d_Wk, d_K, SEQ_LEN, D_MODEL, HEAD_DIM);
    matmul<<<grid1, block1>>>(d_X, d_Wv, d_V, SEQ_LEN, D_MODEL, HEAD_DIM);

    // scores = Q * Kᵗ / sqrt(d_k) and for Kᵗ, just treat K as (HEAD_DIM x SEQ_LEN)
    dim3 grid2(1, SEQ_LEN);
    dim3 block2(SEQ_LEN, 1);
    matmul<<<grid2, block2>>>(d_Q, d_K, d_scores, SEQ_LEN, HEAD_DIM, SEQ_LEN);

    float scale = sqrtf((float)HEAD_DIM);
    scale<<<1, SEQ_LEN * SEQ_LEN>>>(d_scores, SEQ_LEN * SEQ_LEN, scale);

    // softmax(scores)
    row_softmax<<<SEQ_LEN, 1>>>(d_scores, SEQ_LEN, SEQ_LEN);

    // softmax * V
    matmul<<<grid1, block1>>>(d_scores, d_V, d_out, SEQ_LEN, SEQ_LEN, HEAD_DIM);

    // Copy results
    float h_Q[SEQ_LEN * HEAD_DIM], h_K[SEQ_LEN * HEAD_DIM], h_V[SEQ_LEN * HEAD_DIM], h_scores[SEQ_LEN * SEQ_LEN], h_final[SEQ_LEN * HEAD_DIM];
    cudaMemcpy(h_Q, d_Q, SEQ_LEN * HEAD_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_K, d_K, SEQ_LEN * HEAD_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V, d_V, SEQ_LEN * HEAD_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scores, d_scores, SEQ_LEN * SEQ_LEN * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final, d_out, SEQ_LEN * HEAD_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    print_matrix("Q", h_Q, SEQ_LEN, HEAD_DIM);
    print_matrix("K", h_K, SEQ_LEN, HEAD_DIM);
    print_matrix("V", h_V, SEQ_LEN, HEAD_DIM);
    print_matrix("Attention scores (after softmax)", h_scores, SEQ_LEN, SEQ_LEN);
    print_matrix("Output (Attended V)", h_final, SEQ_LEN, HEAD_DIM);

    cudaFree(d_X); cudaFree(d_Wq); cudaFree(d_Wk); cudaFree(d_Wv);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_scores); cudaFree(d_softmax); cudaFree(d_out);

    return 0;
}
