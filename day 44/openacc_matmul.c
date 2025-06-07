#include <stdio.h>
#include <stdlib.h>

#define N 512

void initialize(float A[N][N], float B[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = 1.0f;
            B[i][j] = (i == j) ? 1.0f : 0.0f; 
        }
}

void print_sample(float C[N][N]) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++)
            printf("%4.1f ", C[i][j]);
        printf("\n");
    }
}

int main() {
    float A[N][N], B[N][N], C[N][N] = {0};
    initialize(A, B);

    #pragma acc parallel loop collapse(2) copyin(A, B) copyout(C)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    printf("Top-left 8x8 block of result matrix C:\n");
    print_sample(C);

    return 0;
}
