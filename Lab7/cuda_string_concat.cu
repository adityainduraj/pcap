#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// CUDA kernel to concatenate the input string N times
__global__ void concatenateString(const char *Sin, char *Sout, int M, int N) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the index is within the length of the input string (M)
    if (idx < M) {
        // Each thread copies the character at position idx in Sin to N positions in Sout
        char ch = Sin[idx];
        for (int k = 0; k < N; k++) {
            int out_idx = idx + k * M;  // Position in Sout: idx, idx+M, idx+2M, ...
            Sout[out_idx] = ch;
        }
    }
}

int main() {
    char *h_Sin;        // Input string on host
    char *h_Sout;       // Output string on host
    char *d_Sin;        // Input string on device
    char *d_Sout;       // Output string on device
    int M;              // Length of input string
    int N;              // Number of repetitions
    int Sout_length;    // Length of output string

    // Input the string
    char buffer[100];
    printf("Enter the input string Sin: ");
    scanf("%s", buffer);
    M = strlen(buffer);

    // Input the number of repetitions
    printf("Enter the number of repetitions N: ");
    scanf("%d", &N);

    // Calculate the length of the output string
    Sout_length = M * N;

    // Allocate memory for the input string on the host
    h_Sin = (char *)malloc((M + 1) * sizeof(char));
    strcpy(h_Sin, buffer);

    // Allocate memory for the output string on the host
    h_Sout = (char *)malloc((Sout_length + 1) * sizeof(char));

    // Allocate memory on the device
    cudaMalloc(&d_Sin, (M + 1) * sizeof(char));
    cudaMalloc(&d_Sout, (Sout_length + 1) * sizeof(char));

    // Copy the input string from host to device
    cudaMemcpy(d_Sin, h_Sin, (M + 1) * sizeof(char), cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    concatenateString<<<blocksPerGrid, threadsPerBlock>>>(d_Sin, d_Sout, M, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy the result back to the host
    cudaMemcpy(h_Sout, d_Sout, Sout_length * sizeof(char), cudaMemcpyDeviceToHost);
    h_Sout[Sout_length] = '\0';  // Null-terminate the output string

    // Print the result
    printf("Input string Sin: %s\n", h_Sin);
    printf("Number of repetitions N: %d\n", N);
    printf("Output string Sout: %s\n", h_Sout);

    // Free device memory
    cudaFree(d_Sin);
    cudaFree(d_Sout);

    // Free host memory
    free(h_Sin);
    free(h_Sout);

    return 0;
}