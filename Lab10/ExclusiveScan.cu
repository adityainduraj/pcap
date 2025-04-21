#include <stdio.h>
#include <cuda.h>

#define N 16  // You can change this value as needed

__global__ void exclusive_scan_blelloch(int* data, int* output, int n) {
    extern __shared__ int temp[]; // allocated on invocation
    int thid = threadIdx.x;

    if (2 * thid < n)     temp[2 * thid] = data[2 * thid];
    else                  temp[2 * thid] = 0;

    if (2 * thid + 1 < n) temp[2 * thid + 1] = data[2 * thid + 1];
    else                  temp[2 * thid + 1] = 0;

    int offset = 1;

    // Up-sweep (reduce) phase
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Clear the last element
    if (thid == 0) {
        temp[n - 1] = 0;
    }

    // Down-sweep phase
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    if (2 * thid < n)     output[2 * thid] = temp[2 * thid];
    if (2 * thid + 1 < n) output[2 * thid + 1] = temp[2 * thid + 1];
}

int main() {
    int h_input[N], h_output[N];

    // Initialize input
    printf("Input:\n");
    for (int i = 0; i < N; i++) {
        h_input[i] = i + 1;
        printf("%d ", h_input[i]);
    }
    printf("\n");

    int* d_input;
    int* d_output;
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, N * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    exclusive_scan_blelloch<<<1, N / 2, N * sizeof(int)>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Exclusive Scan Output:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
