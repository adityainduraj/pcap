#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size, str_length, chunk_size;
    char *S1 = NULL, *S2 = NULL, *local_S1, *local_S2, *local_result, *result = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Read the two strings
        char buffer1[256], buffer2[256];
        printf("Enter first string (S1): ");
        scanf("%s", buffer1);
        printf("Enter second string (S2): ");
        scanf("%s", buffer2);

        str_length = strlen(buffer1);
        
        // Ensure both strings have the same length
        if (str_length != strlen(buffer2)) {
            printf("Error: Strings must be of the same length.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Ensure string length is divisible by the number of processes
        if (str_length % size != 0) {
            printf("Error: String length must be evenly divisible by %d processes.\n", size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        S1 = buffer1;
        S2 = buffer2;
        result = (char *)malloc(2 * str_length * sizeof(char));
    }

    // Broadcast the string length to all processes
    MPI_Bcast(&str_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the chunk size each process will handle
    chunk_size = str_length / size;

    // Allocate space for each process's part
    local_S1 = (char *)malloc(chunk_size * sizeof(char));
    local_S2 = (char *)malloc(chunk_size * sizeof(char));
    local_result = (char *)malloc(2 * chunk_size * sizeof(char));

    // Scatter S1 and S2 to all processes
    MPI_Scatter(S1, chunk_size, MPI_CHAR, local_S1, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(S2, chunk_size, MPI_CHAR, local_S2, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Interleave the characters from local S1 and S2
    for (int i = 0; i < chunk_size; i++) {
        local_result[2 * i] = local_S1[i];
        local_result[2 * i + 1] = local_S2[i];
    }

    // Gather the interleaved result at root
    MPI_Gather(local_result, 2 * chunk_size, MPI_CHAR, result, 2 * chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Root process prints the final result
    if (rank == 0) {
        result[2 * str_length] = '\0';  // Null-terminate the string
        printf("\nResultant String: %s\n", result);
        free(result);
    }

    // Free allocated memory
    free(local_S1);
    free(local_S2);
    free(local_result);

    MPI_Finalize();
    return 0;
}

// mpicc -o mpi_str_alt_concat mpi_str_alt_concat.c
// mpirun -np 2 ./mpi_str_alt_concat
