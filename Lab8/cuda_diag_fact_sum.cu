#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define N 3  // Matrix size
__device__ int factorial(int num) {
    if (num == 0 || num == 1) return 1;
    int fact = 1;
    for (int i = 2; i <= num; i++)
        fact *= i;
    return fact;
}
__device__ int sumOfDigits(int num) {
    int sum = 0;
    while (num > 0) {
        sum += num % 10;
        num /= 10;
    }
    return sum;
}
__global__ void transformMatrix(int *A, int *B, int size) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < size && col < size) {
        int value = A[row * size + col];
        if (row == col) {
            B[row * size + col] = 0;  
        } else if (row < col) {
            B[row * size + col] = factorial(value);  
        } else {
            B[row * size + col] = sumOfDigits(value);  
        }
    }
}
void printMatrix(int *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d\t", matrix[i * size + j]);
        }
        printf("\n");
    }
}
int main() {
    int A[N * N], B[N * N];
    int *d_A, *d_B;
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 10;  
    }
    printf("Original Matrix A:\n");
    printMatrix(A, N);
    cudaMalloc(&d_A, N * N * sizeof(int));
    cudaMalloc(&d_B, N * N * sizeof(int));
    cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    transformMatrix<<<N, N>>>(d_A, d_B, N);
    cudaMemcpy(B, d_B, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nTransformed Matrix B:\n");
    printMatrix(B, N);
    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}
