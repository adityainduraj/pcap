#include <stdio.h>
#include <cuda_runtime.h>

#define N 16  // Input size

__global__ void inclusiveScanKernel(int *data, int width) {
    int tid = threadIdx.x;
    
    for (int offset = 1; offset < width; offset *= 2) {
        int temp = (tid >= offset) ? data[tid - offset] : 0;
        __syncthreads();
        if (tid >= offset) {
            data[tid] += temp;
        }
        __syncthreads();
    }
}

void inclusiveScan(int *input, int *output, int width) {
    int *d_data;
    int size = width * sizeof(int);
    
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, input, size, cudaMemcpyHostToDevice);
    
    inclusiveScanKernel<<<1, width>>>(d_data, width);
    
    cudaMemcpy(output, d_data, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
}

int main() {
    int input[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int output[N];

    inclusiveScan(input, output, N);

    printf("Inclusive Scan Result:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    return 0;
}
