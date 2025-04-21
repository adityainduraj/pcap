#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define WIDTH 10
#define MASK_WIDTH 3

__global__ void convolution1D(int *N, int *M, int *P, int width, int mask_width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = mask_width / 2;
    int sum = 0;
    
    if (i < width) {
        for (int j = -radius; j <= radius; j++) {
            int index = i + j;
            if (index >= 0 && index < width) {
                sum += N[index] * M[j + radius];
            }
        }
        P[i] = sum;
    }
}

void printArray(int *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

int main() {
    int h_N[WIDTH] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int h_M[MASK_WIDTH] = {1, 0, -1};
    int h_P[WIDTH];
    
    int *d_N, *d_M, *d_P;
    size_t size_N = WIDTH * sizeof(int);
    size_t size_M = MASK_WIDTH * sizeof(int);
    
    cudaMalloc((void**)&d_N, size_N);
    cudaMalloc((void**)&d_M, size_M);
    cudaMalloc((void**)&d_P, size_N);
    
    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, size_M, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (WIDTH + threadsPerBlock - 1) / threadsPerBlock;
    
    convolution1D<<<blocksPerGrid, threadsPerBlock>>>(d_N, d_M, d_P, WIDTH, MASK_WIDTH);
    
    cudaMemcpy(h_P, d_P, size_N, cudaMemcpyDeviceToHost);
    
    printf("Resultant Array:\n");
    printArray(h_P, WIDTH);
    
    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);
    
    return 0;
}
