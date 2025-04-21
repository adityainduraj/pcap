#include <stdio.h>
#include <cuda_runtime.h>

__global__ void spmv_kernel(int *csrVal, int *csrColIdx, int *csrRowPtr, int *x, int *y, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        int start = csrRowPtr[row];
        int end = csrRowPtr[row + 1];
        int sum = 0;
        for (int i = start; i < end; i++) {
            sum += csrVal[i] * x[csrColIdx[i]];
        }
        y[row] = sum;
    }
}

void spmv(int *csrVal, int *csrColIdx, int *csrRowPtr, int *x, int *y, int N, int nnz) {
    int *d_csrVal, *d_csrColIdx, *d_csrRowPtr, *d_x, *d_y;

    cudaMalloc((void**)&d_csrVal, nnz * sizeof(int));
    cudaMalloc((void**)&d_csrColIdx, nnz * sizeof(int));
    cudaMalloc((void**)&d_csrRowPtr, (N + 1) * sizeof(int));
    cudaMalloc((void**)&d_x, N * sizeof(int));
    cudaMalloc((void**)&d_y, N * sizeof(int));

    cudaMemcpy(d_csrVal, csrVal, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdx, csrColIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrRowPtr, csrRowPtr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    spmv_kernel<<<numBlocks, blockSize>>>(d_csrVal, d_csrColIdx, d_csrRowPtr, d_x, d_y, N);

    cudaMemcpy(y, d_y, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_csrVal);
    cudaFree(d_csrColIdx);
    cudaFree(d_csrRowPtr);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    int N = 4; 
    int nnz = 6; 
    int csrVal[] = {10, 20, 30, 40, 50, 60};
    int csrColIdx[] = {0, 1, 2, 2, 3, 3};
    int csrRowPtr[] = {0, 2, 4, 5, 6};
    int x[] = {1, 2, 3, 4};
    int y[N];
    spmv(csrVal, csrColIdx, csrRowPtr, x, y, N, nnz);
    for (int i = 0; i < N; i++) {
        printf("%d ", y[i]);
    }
    return 0;
}
