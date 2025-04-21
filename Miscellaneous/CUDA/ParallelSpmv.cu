#include <stdio.h>
#include <stdlib.h>

#define N 4  // Number of rows (can be changed)
#define NNZ 9  // Number of non-zero elements (based on matrix below)

// CUDA kernel for SpMV using CSR
__global__ void spmv_csr_kernel(int *row_ptr, int *col_ind, float *val, float *x, float *y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        for (int i = row_start; i < row_end; i++) {
            sum += val[i] * x[col_ind[i]];
        }
        y[row] = sum;
    }
}

int main() {
    // Example matrix (4x4):
    // [10 0 0 0]
    // [0 20 0 0]
    // [30 0 30 0]
    // [0 40 0 40]

    // CSR representation
    int h_row_ptr[N + 1] = {0, 1, 2, 4, 6};
    int h_col_ind[NNZ] = {0, 1, 0, 2, 1, 3};
    float h_val[NNZ] = {10, 20, 30, 30, 40, 40};
    float h_x[N] = {1, 2, 3, 4}; // input vector
    float h_y[N] = {0};          // output vector

    // Device memory
    int *d_row_ptr, *d_col_ind;
    float *d_val, *d_x, *d_y;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_ind, NNZ * sizeof(int));
    cudaMalloc((void**)&d_val, NNZ * sizeof(float));
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_row_ptr, h_row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, h_col_ind, NNZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, h_val, NNZ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    spmv_csr_kernel<<<gridSize, blockSize>>>(d_row_ptr, d_col_ind, d_val, d_x, d_y, N);

    // Copy result back to host
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Result of SpMV:\n");
    for (int i = 0; i < N; i++) {
        printf("y[%d] = %.2f\n", i, h_y[i]);
    }

    // Free device memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
