#include <stdio.h>
#include <stdlib.h>

#define MAX_WIDTH 100
#define MAX_MASK_WIDTH 10

// CUDA Kernel for 1D convolution
__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float Pvalue = 0;
    int N_start_point = i - (Mask_Width / 2);

    for (int j = 0; j < Mask_Width; j++) {
        if ((N_start_point + j >= 0) && (N_start_point + j < Width)) {
            Pvalue += N[N_start_point + j] * M[j];
        }
    }

    if (i < Width) {
        P[i] = Pvalue;
    }
}

int main() {
    int Width, Mask_Width;

    printf("Enter the size of the input array (<= %d): ", MAX_WIDTH);
    scanf("%d", &Width);

    float h_N[MAX_WIDTH], h_M[MAX_MASK_WIDTH], h_P[MAX_WIDTH];

    printf("Enter input array elements:\n");
    for (int i = 0; i < Width; i++) {
        scanf("%f", &h_N[i]);
    }

    printf("Enter the size of the mask (odd number <= %d): ", MAX_MASK_WIDTH);
    scanf("%d", &Mask_Width);

    printf("Enter mask elements:\n");
    for (int i = 0; i < Mask_Width; i++) {
        scanf("%f", &h_M[i]);
    }

    // Device pointers
    float *d_N, *d_M, *d_P;

    // Allocate device memory
    cudaMalloc((void**)&d_N, Width * sizeof(float));
    cudaMalloc((void**)&d_M, Mask_Width * sizeof(float));
    cudaMalloc((void**)&d_P, Width * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_N, h_N, Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, Mask_Width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (Width + blockSize - 1) / blockSize;
    convolution_1D_basic_kernel<<<gridSize, blockSize>>>(d_N, d_M, d_P, Mask_Width, Width);

    // Copy result back to host
    cudaMemcpy(h_P, d_P, Width * sizeof(float), cudaMemcpyDeviceToHost);

    // Display result
    printf("Output array after 1D convolution:\n");
    for (int i = 0; i < Width; i++) {
        printf("%.2f ", h_P[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    return 0;
}
