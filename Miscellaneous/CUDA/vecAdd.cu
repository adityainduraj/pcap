#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define n 1024  // Vector size

__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n)
        C[i] = A[i] + B[i];
}

int main(void) {
    float *A, *B, *C;           // Host arrays
    float *d_A, *d_B, *d_C;     // Device arrays
    int size = n * sizeof(float);

    // Allocate memory on host
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // Initialize A and B
    for (int i = 0; i < n; i++) {
        A[i] = i * 1.0f;
        B[i] = (n - i) * 1.0f;
    }

    // Part-1: Allocate memory on device
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Part-2: Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (int)ceil((float)n / threadsPerBlock);
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    // Part-3: Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print part of result for verification
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, C[i]);
    }

    // Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);

    return 0;
}
