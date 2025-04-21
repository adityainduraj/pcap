#include <stdio.h>
#include <cuda_runtime.h>

__global__ void calculateTotal(int* prices, int* quantities, int* total, int numItems) {
    int idx = threadIdx.x;
    if (idx < numItems) {
        atomicAdd(total, prices[idx] * quantities[idx]);
    }
}

int main() {
    int numItems;
    printf("Enter the number of items: ");
    scanf("%d", &numItems);

    int prices[numItems];
    int quantities[numItems];

    printf("Enter the prices of the items:\n");
    for (int i = 0; i < numItems; ++i) {
        scanf("%d", &prices[i]);
    }

    printf("Enter the quantities purchased by friends:\n");
    for (int i = 0; i < numItems; ++i) {
        scanf("%d", &quantities[i]);
    }

    int* d_prices;
    int* d_quantities;
    int* d_total;
    int total = 0;

    cudaMalloc(&d_prices, numItems * sizeof(int));
    cudaMalloc(&d_quantities, numItems * sizeof(int));
    cudaMalloc(&d_total, sizeof(int));

    cudaMemcpy(d_prices, prices, numItems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_quantities, quantities, numItems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_total, &total, sizeof(int), cudaMemcpyHostToDevice);

    calculateTotal<<<1, numItems>>>(d_prices, d_quantities, d_total, numItems);

    cudaMemcpy(&total, d_total, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Total purchase done by friends: %d\n", total);

    cudaFree(d_prices);
    cudaFree(d_quantities);
    cudaFree(d_total);

    return 0;
}

