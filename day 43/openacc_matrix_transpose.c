#include <stdio.h>
#include <stdlib.h>

#define N 512

void initialize(float A[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = i * N + j;
}

void print_matrix(float A[N][N]) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++)
            printf("%6.1f ", A[i][j]);
        printf("\n");
    }
}

int main() {
    float A[N][N], B[N][N];
    initialize(A);

    #pragma acc parallel loop collapse(2) copyin(A) copyout(B)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[j][i] = A[i][j];
        }
    }

    printf("Sample of Transposed Matrix B:\n");
    print_matrix(B);

    return 0;
}
