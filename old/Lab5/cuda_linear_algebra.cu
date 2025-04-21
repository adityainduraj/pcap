#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel to perform y = alpha * x + y
__global__ void saxpy(int n, float alpha, float *x, float *y) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the index is within the vector size
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

int main() {
    int n;              // Size of the vectors
    float alpha;        // Scalar value
    float *h_x, *h_y;   // Host vectors
    float *d_x, *d_y;   // Device vectors

    // Input the size of the vectors
    printf("Enter the size of the vectors: ");
    scanf("%d", &n);

    // Input the scalar alpha
    printf("Enter the scalar alpha: ");
    scanf("%f", &alpha);

    // Allocate memory for host vectors
    h_x = (float *)malloc(n * sizeof(float));
    h_y = (float *)malloc(n * sizeof(float));

    // Input the elements of vector x
    printf("Enter the elements of vector x:\n");
    for (int i = 0; i < n; i++) {
        scanf("%f", &h_x[i]);
    }

    // Input the elements of vector y
    printf("Enter the elements of vector y:\n");
    for (int i = 0; i < n; i++) {
        scanf("%f", &h_y[i]);
    }

    // Allocate memory for device vectors
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    // Copy vectors from host to device
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the SAXPY kernel
    saxpy<<<blocksPerGrid, threadsPerBlock>>>(n, alpha, d_x, d_y);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy the result back to the host
    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("\nResulting vector y (after y = alpha * x + y):\n");
    for (int i = 0; i < n; i++) {
        printf("%.2f ", h_y[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);

    // Free host memory
    free(h_x);
    free(h_y);

    return 0;
}