#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Function to check if a character is a vowel
int is_vowel(char c) {
    c = tolower(c);
    return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u');
}

int main(int argc, char *argv[]) {
    int rank, size, str_length, local_count = 0, total_non_vowels = 0;
    char *str = NULL;
    char *local_str;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Read the input string
        printf("Enter the string: ");
        fflush(stdout);
        char buffer[256];
        scanf("%s", buffer);

        str_length = strlen(buffer);

        // Ensure the string length is divisible by the number of processes
        if (str_length % size != 0) {
            printf("Error: String length must be evenly divisible by %d processes.\n", size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        str = buffer;
    }

    // Broadcast the string length to all processes
    MPI_Bcast(&str_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process gets str_length / size characters
    int chunk_size = str_length / size;
    local_str = (char *)malloc(chunk_size * sizeof(char));

    // Scatter the string to all processes
    MPI_Scatter(str, chunk_size, MPI_CHAR, local_str, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Count non-vowel characters in the received chunk
    for (int i = 0; i < chunk_size; i++) {
        if (!is_vowel(local_str[i])) {
            local_count++;
        }
    }

    // Gather all local counts at the root process
    int *counts = NULL;
    if (rank == 0) {
        counts = (int *)malloc(size * sizeof(int));
    }

    MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Print results at the root process
    if (rank == 0) {
        printf("\nNon-vowel counts from each process:\n");
        for (int i = 0; i < size; i++) {
            printf("Process %d: %d non-vowels\n", i, counts[i]);
            total_non_vowels += counts[i];
        }
        printf("Total non-vowel count: %d\n", total_non_vowels);
        free(counts);
    }

    free(local_str);
    MPI_Finalize();
    return 0;
}

// mpicc -o mpi_non-vowel_count mpi_non-vowel_count.c
// mpirun -np 4 ./mpi_non-vowel_count
