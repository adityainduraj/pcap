#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define MAX_LEN 100

int main(int argc, char** argv) {
    int rank, size, N;
    char word[MAX_LEN];
    char subword[MAX_LEN];
    char all_subwords[MAX_LEN * MAX_LEN];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Hardcoded input word (can be changed to scanf if desired)
        strcpy(word, "DAVE");
        N = strlen(word);

        if (size != N) {
            printf("Error: Number of processes must be equal to length of word (%d)\n", N);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast word and its length to all processes
    MPI_Bcast(word, MAX_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process computes its part: word[0 to N-rank-1]
    strncpy(subword, word, N - rank);
    subword[N - rank] = '\0';

    // Gather substrings to root
    MPI_Gather(subword, MAX_LEN, MPI_CHAR, all_subwords, MAX_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Output: ");
        for (int i = 0; i < N; i++) {
            printf("%s", &all_subwords[i * MAX_LEN]);
            if (i != N - 1) printf("");
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}

