#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define MAX_LEN 256

__global__ void expandString(char *input, char *output, int *positions, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int startPos = positions[i];
        for (int j = 0; j < i + 1; j++) {
            output[startPos + j] = input[i];
        }
    }
}

void computePositions(int *positions, int length) {
    positions[0] = 0;
    for (int i = 1; i < length; i++) {
        positions[i] = positions[i - 1] + i;
    }
}

int main() {
    char h_input[MAX_LEN] = "Hai";
    int length = strlen(h_input);
    
    int h_positions[MAX_LEN];
    computePositions(h_positions, length);
    int outputLength = h_positions[length - 1] + length;
    char h_output[MAX_LEN] = {0};
    
    char *d_input, *d_output;
    int *d_positions;
    
    cudaMalloc((void**)&d_input, length + 1);
    cudaMalloc((void**)&d_output, outputLength + 1);
    cudaMalloc((void**)&d_positions, length * sizeof(int));
    
    cudaMemcpy(d_input, h_input, length + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions, length * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;
    expandString<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_positions, length);
    
    cudaMemcpy(h_output, d_output, outputLength + 1, cudaMemcpyDeviceToHost);
    
    printf("Expanded Output: %s\n", h_output);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_positions);
    
    return 0;
}
