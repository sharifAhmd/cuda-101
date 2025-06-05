#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512
#define STEPS 1000

float A[N][N];
float B[N][N];

void initialize() {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A[i][j] = (i == 0 || j == 0 || i == N-1 || j == N-1) ? 100.0f : 0.0f;  // Hot edges
}

int main() {
    initialize();

    for (int step = 0; step < STEPS; ++step) {
        #pragma acc parallel loop collapse(2) copyin(A) copyout(B)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                B[i][j] = 0.25f * (A[i+1][j] + A[i-1][j] + A[i][j+1] + A[i][j-1]);
            }
        }

        #pragma acc parallel loop collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                A[i][j] = B[i][j];
            }
        }
    }


    printf("Center temperature after %d steps: %.2f\n", STEPS, A[N/2][N/2]);

    return 0;
}
