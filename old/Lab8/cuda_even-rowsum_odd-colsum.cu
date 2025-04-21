#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define M 3  
#define N 3  
__global__ void computeRowSum(int *A, int *rowSum, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        int sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += A[row * cols + j];
        }
        rowSum[row] = sum;
    }
}
__global__ void computeColumnSum(int *A, int *colSum, int rows, int cols) {
    int col = blockIdx.x;
    if (col < cols) {
        int sum = 0;
        for (int i = 0; i < rows; i++) {
            sum += A[i * cols + col];
        }
        colSum[col] = sum;
    }
}
__global__ void transformMatrix(int *A, int *B, int *rowSum, int *colSum, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < rows && col < cols) {
        int value = A[row * cols + col];
        if (value % 2 == 0) {
            B[row * cols + col] = rowSum[row];  
        } else {
            B[row * cols + col] = colSum[col];  
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
    int A[M * N], B[M * N];
    int *d_A, *d_B, *d_rowSum, *d_colSum;
    for (int i = 0; i < M * N; i++) {
        A[i] = rand() % 10;
    }
    printf("Original Matrix A:\n");
    printMatrix(A, M, N);
    cudaMalloc(&d_A, M * N * sizeof(int));
    cudaMalloc(&d_B, M * N * sizeof(int));
    cudaMalloc(&d_rowSum, M * sizeof(int));
    cudaMalloc(&d_colSum, N * sizeof(int));
    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);
    computeRowSum<<<M, 1>>>(d_A, d_rowSum, M, N);
    computeColumnSum<<<N, 1>>>(d_A, d_colSum, M, N);
    transformMatrix<<<M, N>>>(d_A, d_B, d_rowSum, d_colSum, M, N);
    cudaMemcpy(B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nTransformed Matrix B:\n");
    printMatrix(B, M, N);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_rowSum);
    cudaFree(d_colSum);
    return 0;
}
