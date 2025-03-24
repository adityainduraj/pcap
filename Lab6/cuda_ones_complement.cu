#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 10

__device__ int onesComplement(int num) {
    int complement = 0, place = 1;
    while (num > 0) {
        int bit = num % 10;
        complement += ((bit == 0) ? 1 : 0) * place;
        num /= 10;
        place *= 10;
    }
    return complement;
}

__global__ void computeOnesComplement(int *input, int *output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = onesComplement(input[i]);
    }
}

void printArray(int *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

int main() {
    int h_input[N] = {1010, 1100, 1001, 1111, 0000, 1011, 1101, 1000, 0110, 0011};
    int h_output[N];
    
    int *d_input, *d_output;
    size_t size = N * sizeof(int);
    
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    computeOnesComplement<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    printf("1's Complement Values:\n");
    printArray(h_output, N);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
