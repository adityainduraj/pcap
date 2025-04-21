#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define M 2
#define N 4

__global__ void buildStringKernel(char* A, int* B, char* STR, int* offsetTable) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        char ch = A[idx];
        int count = B[idx];
        int offset = offsetTable[idx];

        for (int k = 0; k < count; ++k) {
            STR[offset + k] = ch;
        }
    }
}

int main() {
    // Host matrices
    char h_A[M * N] = {
        'p', 'C', 'a', 'P',
        'e', 'X', 'a', 'M'
    };
    int h_B[M * N] = {
        1, 2, 4, 3,
        2, 4, 3, 2
    };

    // Compute total output size and offsets
    int h_offset[M * N];
    int totalLen = 0;
    for (int i = 0; i < M * N; ++i) {
        h_offset[i] = totalLen;
        totalLen += h_B[i];
    }

    // Allocate device memory
    char *d_A, *d_STR;
    int *d_B, *d_offset;
    cudaMalloc((void**)&d_A, M * N * sizeof(char));
    cudaMalloc((void**)&d_B, M * N * sizeof(int));
    cudaMalloc((void**)&d_offset, M * N * sizeof(int));
    cudaMalloc((void**)&d_STR, totalLen * sizeof(char));

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, M * N * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset, h_offset, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (M * N + threadsPerBlock - 1) / threadsPerBlock;
    buildStringKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_STR, d_offset);

    // Copy result back
    char* h_STR = (char*)malloc(totalLen + 1);
    cudaMemcpy(h_STR, d_STR, totalLen * sizeof(char), cudaMemcpyDeviceToHost);
    h_STR[totalLen] = '\0'; // null-terminate the string

    // Print the output
    printf("Output string - STR: %s\n", h_STR);

    // Clean up
    free(h_STR);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_offset);
    cudaFree(d_STR);

    return 0;
}

