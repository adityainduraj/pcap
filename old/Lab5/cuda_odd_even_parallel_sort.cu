#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 4 // Number of rows
#define M 5 // Number of columns

__global__ void oddEvenSortRows(int *matrix, int cols) {
    int row = blockIdx.x;
    for (int phase = 0; phase < cols; phase++) {
        int i = threadIdx.x * 2 + (phase % 2);
        if (i < cols - 1) {
            int idx1 = row * cols + i;
            int idx2 = row * cols + i + 1;
            if (matrix[idx1] > matrix[idx2]) {
                int temp = matrix[idx1];
                matrix[idx1] = matrix[idx2];
                matrix[idx2] = temp;
            }
        }
        __syncthreads();
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
    int h_matrix[N][M] = {
        {64, 34, 25, 12, 22},
        {90, 11, 45, 2, 8},
        {15, 3, 99, 20, 4},
        {78, 56, 30, 10, 18}
    };

    int *d_matrix;
    size_t size = N * M * sizeof(int);
    
    cudaMalloc((void**)&d_matrix, size);
    cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);
    
    oddEvenSortRows<<<N, M/2>>>(d_matrix, M);
    cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    
    printf("Sorted Matrix:\n");
    printMatrix((int*)h_matrix, N, M);
    
    cudaFree(d_matrix);
    return 0;
}
