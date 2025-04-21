#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 4 // Number of rows
#define M 5 // Number of columns

__global__ void selectionSortRows(int *matrix, int cols) {
    int row = blockIdx.x; // Each block handles one row
    
    for (int i = 0; i < cols - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < cols; j++) {
            if (matrix[row * cols + j] < matrix[row * cols + min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            int temp = matrix[row * cols + i];
            matrix[row * cols + i] = matrix[row * cols + min_idx];
            matrix[row * cols + min_idx] = temp;
        }
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
    
    selectionSortRows<<<N, 1>>>(d_matrix, M);
    cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    
    printf("Sorted Matrix:\n");
    printMatrix((int*)h_matrix, N, M);
    
    cudaFree(d_matrix);
    return 0;
}
