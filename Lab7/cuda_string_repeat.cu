#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// CUDA kernel to copy characters from S to RS
__global__ void repeatString(const char *S, char *RS, int N) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the index is within the output string length (2N)
    if (idx < 2 * N) {
        // Map the output index to the input string index
        RS[idx] = S[idx % N];
    }
}

int main() {
    char *h_S;          // Input string on host
    char *h_RS;         // Output string on host
    char *d_S;          // Input string on device
    char *d_RS;         // Output string on device
    int N;              // Length of input string
    int RS_length;      // Length of output string

    // Input the string
    char buffer[100];
    printf("Enter the input string S: ");
    scanf("%s", buffer);
    N = strlen(buffer);

    // Calculate the length of the output string
    RS_length = 2 * N;

    // Allocate memory for the input string on the host
    h_S = (char *)malloc((N + 1) * sizeof(char));
    strcpy(h_S, buffer);

    // Allocate memory for the output string on the host
    h_RS = (char *)malloc((RS_length + 1) * sizeof(char));

    // Allocate memory on the device
    cudaMalloc(&d_S, (N + 1) * sizeof(char));
    cudaMalloc(&d_RS, (RS_length + 1) * sizeof(char));

    // Copy the input string from host to device
    cudaMemcpy(d_S, h_S, (N + 1) * sizeof(char), cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (RS_length + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    repeatString<<<blocksPerGrid, threadsPerBlock>>>(d_S, d_RS, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy the result back to the host
    cudaMemcpy(h_RS, d_RS, RS_length * sizeof(char), cudaMemcpyDeviceToHost);
    h_RS[RS_length] = '\0';  // Null-terminate the output string

    // Print the result
    printf("Input string S: %s\n", h_S);
    printf("Output string RS: %s\n", h_RS);

    // Free device memory
    cudaFree(d_S);
    cudaFree(d_RS);

    // Free host memory
    free(h_S);
    free(h_RS);

    return 0;
}