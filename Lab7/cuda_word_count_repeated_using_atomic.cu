#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define MAX_WORDS 100
#define MAX_WORD_LENGTH 20

__device__ bool isWordMatch(const char *sentence, int start, const char *word, int wordLen, int sentenceLen) {
    if (start + wordLen > sentenceLen) return false;
    for (int i = 0; i < wordLen; i++) {
        if (sentence[start + i] != word[i]) return false;
    }
    return (start + wordLen == sentenceLen || sentence[start + wordLen] == ' ');
}

__global__ void countWordOccurrences(char *sentence, char *word, int *count, int sentenceLen, int wordLen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sentenceLen && (i == 0 || sentence[i - 1] == ' ')) {
        if (isWordMatch(sentence, i, word, wordLen, sentenceLen)) {
            atomicAdd(count, 1);
        }
    }
}

int main() {
    char h_sentence[] = "cuda is fast and cuda is powerful and cuda is parallel";
    char h_word[] = "cuda";
    int h_count = 0;

    char *d_sentence, *d_word;
    int *d_count;
    int sentenceLen = strlen(h_sentence);
    int wordLen = strlen(h_word);

    cudaMalloc((void**)&d_sentence, sentenceLen + 1);
    cudaMalloc((void**)&d_word, wordLen + 1);
    cudaMalloc((void**)&d_count, sizeof(int));

    cudaMemcpy(d_sentence, h_sentence, sentenceLen + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, h_word, wordLen + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (sentenceLen + threadsPerBlock - 1) / threadsPerBlock;
    countWordOccurrences<<<blocksPerGrid, threadsPerBlock>>>(d_sentence, d_word, d_count, sentenceLen, wordLen);

    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("The word '%s' appears %d times in the sentence.\n", h_word, h_count);

    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_count);

    return 0;
}
