#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void tiledConvolution1D(float* N, float* M, float* P, int width, int mask_width) {
    __shared__ float N_shared[TILE_WIDTH + mask_width - 1];
    int tx = threadIdx.x;
    int start = blockIdx.x * TILE_WIDTH;
    int end = start + TILE_WIDTH + mask_width - 1;

    // Load elements into shared memory
    int load_idx = start + tx - mask_width / 2;
    if (load_idx >= 0 && load_idx < width) {
        N_shared[tx] = N[load_idx];
    } else {
        N_shared[tx] = 0.0f; // Handle boundary conditions
    }

    __syncthreads();

    // Perform convolution
    float result = 0.0f;
    if (tx < TILE_WIDTH) {
        for (int i = 0; i < mask_width; ++i) {
            result += N_shared[tx + i] * M[i];
        }
        if (start + tx < width) {
            P[start + tx] = result;
        }
    }
}

int main() {
    int width, mask_width;
    printf("Enter the width of the input array: ");
    scanf("%d", &width);
    printf("Enter the width of the mask array: ");
    scanf("%d", &mask_width);

    float N[width];
    float M[mask_width];
    float P[width];

    printf("Enter the elements of the input array:\n");
    for (int i = 0; i < width; ++i) {
        scanf("%f", &N[i]);
    }

    printf("Enter the elements of the mask array:\n");
    for (int i = 0; i < mask_width; ++i) {
        scanf("%f", &M[i]);
    }

    float* d_N;
    float* d_M;
    float* d_P;

    cudaMalloc(&d_N, width * sizeof(float));
    cudaMalloc(&d_M, mask_width * sizeof(float));
    cudaMalloc(&d_P, width * sizeof(float));

    cudaMemcpy(d_N, N, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, mask_width * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    tiledConvolution1D<<<numBlocks, TILE_WIDTH>>>(d_N, d_M, d_P, width, mask_width);

    cudaMemcpy(P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Resultant array:\n");
    for (int i = 0; i < width; ++i) {
        printf("%f ", P[i]);
    }
    printf("\n");

    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    return 0;
}

