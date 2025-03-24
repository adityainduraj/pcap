#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define N 3  
__global__ void matrixAddElementWise(int *A, int *B, int *C, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        C[index] = A[index] + B[index];
    }
}
__global__ void matrixAddRowWise(int *A, int *B, int *C, int rows, int cols) {
    int row = blockIdx.x;

    if (row < rows) {
        for (int col = 0; col < cols; col++) {
            int index = row * cols + col;
            C[index] = A[index] + B[index];
        }
    }
}
__global__ void matrixAddColumnWise(int *A, int *B, int *C, int rows, int cols) {
    int col = blockIdx.x;

    if (col < cols) {
        for (int row = 0; row < rows; row++) {
            int index = row * cols + col;
            C[index] = A[index] + B[index];
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
    cudaMalloc(&d_A, N * N * sizeof(int));
    cudaMalloc(&d_B, N * N * sizeof(int));
    cudaMalloc(&d_C, N * N * sizeof(int));
    cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);
    matrixAddElementWise<<<N, N>>>(d_A, d_B, d_C, N, N);
    cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Element-wise Addition:\n");
    printMatrix(C, N, N);
    matrixAddRowWise<<<N, 1>>>(d_A, d_B, d_C, N, N);
    cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nRow-wise Addition:\n");
    printMatrix(C, N, N);
    matrixAddColumnWise<<<N, 1>>>(d_A, d_B, d_C, N, N);
    cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nColumn-wise Addition:\n");
    printMatrix(C, N, N);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
