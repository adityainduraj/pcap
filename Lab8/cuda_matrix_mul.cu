#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define N 3  
__global__ void matrixMulElementWise(int *A, int *B, int *C, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < cols) {
        int sum = 0;
        for (int k = 0; k < cols; k++) {
            sum += A[row * cols + k] * B[k * cols + col];
        }
        C[row * cols + col] = sum;
    }
}
__global__ void matrixMulRowWise(int *A, int *B, int *C, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        for (int col = 0; col < cols; col++) {
            int sum = 0;
            for (int k = 0; k < cols; k++) {
                sum += A[row * cols + k] * B[k * cols + col];
            }
            C[row * cols + col] = sum;
        }
    }
}
__global__ void matrixMulColumnWise(int *A, int *B, int *C, int rows, int cols) {
    int col = blockIdx.x;

    if (col < cols) {
        for (int row = 0; row < rows; row++) {
            int sum = 0;
            for (int k = 0; k < cols; k++) {
                sum += A[row * cols + k] * B[k * cols + col];
            }
            C[row * cols + col] = sum;
        }
    }
}
void printMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}
int main() {
    int A[N * N], B[N * N], C[N * N];
    int *d_A, *d_B, *d_C;

    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }
    printf("Matrix A:\n");
    printMatrix(A, N, N);
    printf("\nMatrix B:\n");
    printMatrix(B, N, N);
    cudaMalloc(&d_A, N * N * sizeof(int));
    cudaMalloc(&d_B, N * N * sizeof(int));
    cudaMalloc(&d_C, N * N * sizeof(int));
    cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);
    matrixMulElementWise<<<N, N>>>(d_A, d_B, d_C, N, N);
    cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nElement-wise Multiplication:\n");
    printMatrix(C, N, N);
    matrixMulRowWise<<<N, 1>>>(d_A, d_B, d_C, N, N);
    cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nRow-wise Multiplication:\n");
    printMatrix(C, N, N);
    matrixMulColumnWise<<<N, 1>>>(d_A, d_B, d_C, N, N);
    cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nColumn-wise Multiplication:\n");
    printMatrix(C, N, N);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

