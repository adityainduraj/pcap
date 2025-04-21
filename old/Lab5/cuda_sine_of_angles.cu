#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

__global__ void calculateSine(float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        output[i] = sinf(input[i]);
    }
}

int main() {
    int N = 10;
    size_t size = N * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_input[i] = i * 0.1;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    calculateSine<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    printf("Angle(rad):\tSines");
    for (int i = 0; i < N; i++) {
        printf("\nsin(%f):\t%f\n", h_input[i], h_output[i]);
    }
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}

