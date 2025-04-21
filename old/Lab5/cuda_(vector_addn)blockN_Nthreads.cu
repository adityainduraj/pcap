#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vectorAdd_blockSizeAsN(int* A, int* B, int* C, int N) {
    int index = threadIdx.x;
    if (index < N) {
        C[index] = A[index] + B[index];
    }
}

__global__ void vectorAdd_NThreads(int* A, int* B, int* C, int N) {
    int index = blockIdx.x;
    if (index < N) {
        C[index] = A[index] + B[index];
    }
}

int main() {
    int N = 10;
    size_t size = N * sizeof(int);

    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);
    int *h_C = (int*)malloc(size);
    
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    vectorAdd_blockSizeAsN<<<1, N>>>(d_A, d_B, d_C, N);

    vectorAdd_NThreads<<<N, 1>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result Vector C:\n");
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", h_A[i], h_B[i], h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

