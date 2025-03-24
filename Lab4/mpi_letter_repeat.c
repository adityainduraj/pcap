#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char *input_word = NULL;      // Input word at root
    char local_char;              // Character for each process
    char *local_output = NULL;    // Repeated characters for each process
    char *final_output = NULL;    // Final output string at root
    int *recv_counts = NULL;      // Number of characters each process sends
    int *displacements = NULL;    // Displacements for MPI_Gatherv
    int N;                        // Length of the input word

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process reads the input word
    if (rank == 0) {
        char buffer[100];
        printf("Enter a word: ");
        scanf("%s", buffer);
        N = strlen(buffer);

        // Check if the number of processes matches the length of the word
        if (N != size) {
            fprintf(stderr, "Error: Number of processes (%d) must match the length of the word (%d).\n", size, N);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Allocate memory for the input word
        input_word = (char *)malloc((N + 1) * sizeof(char));
        strcpy(input_word, buffer);

        // Print the input word for clarity
        printf("Input word: %s\n", input_word);
    }

    // Broadcast the length of the word to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Check if the number of processes matches the length of the word in all processes
    if (N != size) {
        MPI_Finalize();
        return 1;
    }

    // Scatter the characters of the input word to all processes
    MPI_Scatter(input_word, 1, MPI_CHAR, &local_char, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Each process repeats its character (rank + 1) times
    int repeat_count = rank + 1;
    local_output = (char *)malloc((repeat_count + 1) * sizeof(char));
    for (int i = 0; i < repeat_count; i++) {
        local_output[i] = local_char;
    }
    local_output[repeat_count] = '\0';  // Null-terminate the string

    // Prepare for MPI_Gatherv
    if (rank == 0) {
        // Allocate arrays for receive counts and displacements
        recv_counts = (int *)malloc(size * sizeof(int));
        displacements = (int *)malloc(size * sizeof(int));

        // Compute the total length of the output
        int total_length = (N * (N + 1)) / 2;
        final_output = (char *)malloc((total_length + 1) * sizeof(char));

        // Set receive counts and displacements
        int offset = 0;
        for (int i = 0; i < size; i++) {
            recv_counts[i] = i + 1;  // Process i sends (i+1) characters
            displacements[i] = offset;
            offset += recv_counts[i];
        }
    }

    // Gather the repeated characters from all processes to the root
    MPI_Gatherv(local_output, repeat_count, MPI_CHAR, 
                final_output, recv_counts, displacements, MPI_CHAR, 
                0, MPI_COMM_WORLD);

    // Root process prints the final output
    if (rank == 0) {
        final_output[(N * (N + 1)) / 2] = '\0';  // Null-terminate the final string
        printf("Output word: %s\n", final_output);

        // Free allocated memory
        free(input_word);
        free(recv_counts);
        free(displacements);
        free(final_output);
    }

    // Free local memory
    free(local_output);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

// mpicc -o mpi_word_pattern mpi_word_pattern.c
// mpirun -np 4 ./mpi_word_pattern