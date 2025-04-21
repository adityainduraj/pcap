#include <stdio.h>
#include <cuda_runtime.h>

#define M 2
#define N 3

__global__ void computeRowColSums(int* A, int* rowSum, int* colSum) {
    int tid = threadIdx.x;

    // Compute row sums
    if (tid < M) {
        int sum = 0;
        for (int j = 0; j < N; j++) {
            sum += A[tid * N + j];
        }
        rowSum[tid] = sum;
    }

    // Compute column sums
    if (tid < N) {
        int sum = 0;
        for (int i = 0; i < M; i++) {
            sum += A[i * N + tid];
        }
        colSum[tid] = sum;
    }
}

__global__ void computeOutputMatrix(int* rowSum, int* colSum, int* B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        B[row * N + col] = rowSum[row] + colSum[col];
    }
}

int main() {
    int h_A[M * N] = {
        1, 2, 3,
        4, 5, 6
    };
    int h_B[M * N];

    int *d_A, *d_rowSum, *d_colSum, *d_B;

    cudaMalloc((void**)&d_A, M * N * sizeof(int));
    cudaMalloc((void**)&d_rowSum, M * sizeof(int));
    cudaMalloc((void**)&d_colSum, N * sizeof(int));
    cudaMalloc((void**)&d_B, M * N * sizeof(int));

    cudaMemcpy(d_A, h_A, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // Use max(M, N) threads to safely handle both row and column sums
    int maxDim = (M > N) ? M : N;
    computeRowColSums<<<1, maxDim>>>(d_A, d_rowSum, d_colSum);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);
    computeOutputMatrix<<<gridDim, blockDim>>>(d_rowSum, d_colSum, d_B);

    cudaMemcpy(h_B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Output matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_B[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_rowSum);
    cudaFree(d_colSum);
    cudaFree(d_B);

    return 0;
}

