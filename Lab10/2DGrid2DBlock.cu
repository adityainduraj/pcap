#include <stdio.h>
#include <cuda_runtime.h>

#define N 16  // Matrix size

__global__ void matrixMulKernel(int *A, int *B, int *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        int sum = 0;
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

void matrixMul(int *A, int *B, int *C, int width) {
    int size = width * width * sizeof(int);
    int *d_A, *d_B, *d_C;
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(4, 4);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (width + blockDim.y - 1) / blockDim.y);
    
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, width);
    
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int A[N * N], B[N * N], C[N * N];
    
    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = i % N;
        B[i] = i % N;
    }

    matrixMul(A, B, C, N);

    // Print result
    printf("Result Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", C[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}

