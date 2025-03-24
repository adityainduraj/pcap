#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 10

__device__ int toOctal(int num) {
    int octal = 0, place = 1;
    while (num > 0) {
        octal += (num % 8) * place;
        num /= 8;
        place *= 10;
    }
    return octal;
}

__global__ void convertToOctal(int *input, int *output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = toOctal(input[i]);
    }
}

void printArray(int *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

int main() {
    int h_input[N] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    int h_output[N];
    
    int *d_input, *d_output;
    size_t size = N * sizeof(int);
    
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    convertToOctal<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    printf("Octal Values:\n");
    printArray(h_output, N);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
