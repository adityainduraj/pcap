#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <conio.h>

#define N 1024 

__global__ void CUDACount(char* A, unsigned int *d_count, int len) { 
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < len && A[i] == 'a') { 
        atomicAdd(d_count, 1);
    }
}

int main() { 
    char A[N]; 
    char *d_A; 
    unsigned int count = 0; 
    unsigned int *d_count; 
    unsigned int result = 0;

    printf("Enter a string: "); 
    fgets(A, N, stdin);
    int len = strlen(A);

    // CUDA timing
    cudaEvent_t start, stop; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 
    cudaEventRecord(start, 0); 

    // Allocate memory on device
    cudaMalloc((void**)&d_A, len * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(unsigned int)); 

    // Copy data to device
    cudaMemcpy(d_A, A, len * sizeof(char), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_count, &count, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
    CUDACount<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_count, len); 

    cudaError_t error = cudaGetLastError(); 
    if (error != cudaSuccess) { 
        printf("CUDA Error: %s\n", cudaGetErrorString(error)); 
        return 1;
    } 

    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop); 

    // Measure time
    float elapsedTime; 
    cudaEventElapsedTime(&elapsedTime, start, stop); 

    // Copy result back to host
    cudaMemcpy(&result, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost); 

    printf("Total occurrences of 'a' = %u\n", result); 
    printf("Time Taken = %f ms\n", elapsedTime); 

    // Free memory
    cudaFree(d_A); 
    cudaFree(d_count); 

    getch(); 
    return 0; 
}
