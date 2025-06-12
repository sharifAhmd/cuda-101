#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <math.h>

#define SEQ_LEN 8      
#define D_MODEL 8    
#define HEAD_DIM 8    

// Parallel row-wise softmax kernel with shared memory
__global__ void row_softmax(float* scores, int rows, int cols) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row < rows) {

        float maxval = -1e20f;
        for (int i = tid; i < cols; i += blockDim.x)
            maxval = fmaxf(maxval, scores[row * cols + i]);

        shared[tid] = maxval;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
            __syncthreads();
        }
        maxval = shared[0];

        float sum = 0.0f;
        for (int i = tid; i < cols; i += blockDim.x) {
            scores[row * cols + i] = expf(scores[row * cols + i] - maxval);
            sum += scores[row * cols + i];
        }
        shared[tid] = sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared[tid] += shared[tid + s];
            __syncthreads();
        }
        sum = shared[0];

        for (int i = tid; i < cols; i += blockDim.x)
            scores[row * cols + i] /= sum;
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

    float h_X[SEQ_LEN * D_MODEL];
    for (int i = 0; i < SEQ_LEN * D_MODEL; ++i)
        h_X[i] = (i % D_MODEL) + 1;

    float h_Wq[D_MODEL * HEAD_DIM], h_Wk[D_MODEL * HEAD_DIM], h_Wv[D_MODEL * HEAD_DIM];
    for (int i = 0; i < D_MODEL * HEAD_DIM; ++i) {
        h_Wq[i] = (i % HEAD_DIM == i / D_MODEL) ? 1.0f : 0.0f;
        h_Wk[i] = (i % HEAD_DIM == i / D_MODEL) ? 1.0f : 0.0f;
        h_Wv[i] = (i % HEAD_DIM == i / D_MODEL) ? 1.0f : 0.0f;
    }

    float *d_X, *d_Wq, *d_Wk, *d_Wv, *d_Q, *d_K, *d_V, *d_scores, *d_out;
    cudaMalloc(&d_X,     SEQ_LEN * D_MODEL * sizeof(float));
    cudaMalloc(&d_Wq,    D_MODEL * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_Wk,    D_MODEL * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_Wv,    D_MODEL * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_Q,     SEQ_LEN * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_K,     SEQ_LEN * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_V,     SEQ_LEN * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_scores, SEQ_LEN * SEQ_LEN * sizeof(float));
    cudaMalloc(&d_out,   SEQ_LEN * HEAD_DIM * sizeof(float));

    cudaMemcpy(d_X,  h_X,  SEQ_LEN * D_MODEL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wq, h_Wq, D_MODEL * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wk, h_Wk, D_MODEL * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wv, h_Wv, D_MODEL * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);


    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

 
    cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        HEAD_DIM, SEQ_LEN, D_MODEL,
        &alpha, d_Wq, HEAD_DIM, d_X, D_MODEL, &beta, d_Q, HEAD_DIM);

    cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        HEAD_DIM, SEQ_LEN, D_MODEL,
        &alpha, d_Wk, HEAD_DIM, d_X, D_MODEL, &beta, d_K, HEAD_DIM);

    cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        HEAD_DIM, SEQ_LEN, D_MODEL,
        &alpha, d_Wv, HEAD_DIM, d_X, D_MODEL, &beta, d_V, HEAD_DIM);


    float scale = sqrtf((float)HEAD_DIM);

    cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        SEQ_LEN, SEQ_LEN, HEAD_DIM,
        &alpha, d_Q, HEAD_DIM, d_K, HEAD_DIM, &beta, d_scores, SEQ_LEN);


    int num_scores = SEQ_LEN * SEQ_LEN;
    int threads = 128;
    int blocks = (num_scores + threads - 1) / threads;
    scale<<<blocks, threads>>>(d_scores, num_scores, scale);

    row_softmax<<<SEQ_LEN, threads, threads * sizeof(float)>>>(d_scores, SEQ_LEN, SEQ_LEN);

    // Output = softmax(scores) * V
    // (SEQ_LEN, HEAD_DIM) = (SEQ_LEN, SEQ_LEN) * (SEQ_LEN, HEAD_DIM)
    cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        HEAD_DIM, SEQ_LEN, SEQ_LEN,
        &alpha, d_V, HEAD_DIM, d_scores, SEQ_LEN, &beta, d_out, HEAD_DIM);

    float h_attn[SEQ_LEN * SEQ_LEN], h_out[SEQ_LEN * HEAD_DIM];
    cudaMemcpy(h_attn, d_scores, SEQ_LEN * SEQ_LEN * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out,  d_out,    SEQ_LEN * HEAD_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    print_matrix("Attention weights (softmax)", h_attn, SEQ_LEN, SEQ_LEN);
    print_matrix("Attention Output", h_out, SEQ_LEN, HEAD_DIM);

    cublasDestroy(handle);
    cudaFree(d_X); cudaFree(d_Wq); cudaFree(d_Wk); cudaFree(d_Wv);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_scores); cudaFree(d_out);

    return 0;
}

__global__ void scale(float* mat, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) mat[idx] /= scale;
}
