#include <stdio.h>
#include <cuda_runtime.h>

#define N 16   // Input size
#define K 5    // Kernel size

__constant__ int constKernel[K]; // Constant memory for kernel

__global__ void conv1D(int *input, int *output, int width, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = kernel_size / 2;
    
    if (idx >= width) return;
    
    int sum = 0;
    for (int k = -radius; k <= radius; k++) {
        int in_idx = idx + k;
        if (in_idx >= 0 && in_idx < width) {
            sum += input[in_idx] * constKernel[radius + k];
        }
    }
    output[idx] = sum;
}

void convolution1D(int *input, int *kernel, int *output, int width, int kernel_size) {
    int *d_input, *d_output;
    int size_input = width * sizeof(int);
    int size_output = width * sizeof(int);
    
    cudaMalloc(&d_input, size_input);
    cudaMalloc(&d_output, size_output);
    
    cudaMemcpy(d_input, input, size_input, cudaMemcpyHostToDevice);
    
    // Copy kernel to constant memory
    cudaMemcpyToSymbol(constKernel, kernel, kernel_size * sizeof(int));

    dim3 blockDim(8);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x);
    
    conv1D<<<gridDim, blockDim>>>(d_input, d_output, width, kernel_size);
    
    cudaMemcpy(output, d_output, size_output, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int input[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int kernel[K] = {1, 2, 3, 2, 1};
    int output[N];

    convolution1D(input, kernel, output, N, K);

    printf("Output:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    return 0;
}

