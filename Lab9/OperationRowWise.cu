#include <stdio.h>
#include <cuda_runtime.h>
__global__ void modify_matrix_kernel(int *A, int M, int N) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < M && col < N) {
        if (row == 1) {
            A[row * N + col] = A[row * N + col] * A[row * N + col]; 
        } else if (row == 2) {
            A[row * N + col] = A[row * N + col] * A[row * N + col] * A[row * N + col];         }
    }
}
int main() {
    int M = 3, N = 4;
    int A[12] = {1, 2, 3, 4, 6, 5, 8, 3, 2, 4, 10, 1}; // input matrix
    int *d_A;
    cudaMalloc((void**)&d_A, M * N * sizeof(int));
    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);
    int blockSize = N;
    modify_matrix_kernel<<<M, blockSize>>>(d_A, M, N);
    cudaMemcpy(A, d_A, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Modified Matrix A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", A[i * N + j]);
        }
        printf("\n");
    }
    cudaFree(d_A);
    return 0;
}
