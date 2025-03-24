#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define MAX_LEN 256

__device__ void reverseWord(char *sentence, int start, int end) {
    while (start < end) {
        char temp = sentence[start];
        sentence[start] = sentence[end];
        sentence[end] = temp;
        start++;
        end--;
    }
}

__global__ void reverseWords(char *sentence, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length && (i == 0 || sentence[i - 1] == ' ')) {
        int start = i, end = i;
        while (end < length && sentence[end] != ' ') {
            end++;
        }
        reverseWord(sentence, start, end - 1);
    }
}

int main() {
    char h_sentence[MAX_LEN] = "cuda is powerful and fast";
    int length = strlen(h_sentence);
    
    char *d_sentence;
    cudaMalloc((void**)&d_sentence, length + 1);
    cudaMemcpy(d_sentence, h_sentence, length + 1, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;
    reverseWords<<<blocksPerGrid, threadsPerBlock>>>(d_sentence, length);
    
    cudaMemcpy(h_sentence, d_sentence, length + 1, cudaMemcpyDeviceToHost);
    
    printf("Reversed words: %s\n", h_sentence);
    
    cudaFree(d_sentence);
    return 0;
}
