#include <stdio.h>
#include <cuda_runtime.h>
__global__ void complement_matrix_kernel(int *A, int *B, int M, int N) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row > 0 && row < M - 1 && col > 0 && col < N - 1) {
        B[row * N + col] = ~A[row * N + col]; // 1's complement for non-border elements
    } else {
        B[row * N + col] = A[row * N + col]; // border elements remain same
    }
}
int main() {
    int M = 4, N = 4;
    int A[16] = {1, 2, 3, 4, 6, 5, 8, 3, 2, 4, 10, 1, 9, 1, 2, 5};
    int B[16];
    int *d_A, *d_B;
    cudaMalloc((void**)&d_A, M * N * sizeof(int));
    cudaMalloc((void**)&d_B, M * N * sizeof(int));
    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);
    int blockSize = N;
    complement_matrix_kernel<<<M, blockSize>>>(d_A, d_B, M, N);
    cudaMemcpy(B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Modified Matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", B[i * N + j]);
        }
        printf("\n");
    }
    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}
