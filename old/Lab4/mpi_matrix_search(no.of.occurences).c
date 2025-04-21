#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ROWS 3
#define COLS 3

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[ROWS * COLS]; // 1D array to store the 3x3 matrix
    int local_row[COLS];     // Array to store one row for each process
    int search_element;      // Element to search for
    int local_count = 0;     // Count of occurrences in each process
    int total_count = 0;     // Total count across all processes

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if the number of processes is exactly 3
    if (size != 3) {
        if (rank == 0) {
            fprintf(stderr, "Error: This program requires exactly 3 processes, but %d were provided.\n", size);
        }
        MPI_Finalize();
        return 1;
    }

    // Root process reads the matrix and the element to search for
    if (rank == 0) {
        printf("Enter the elements of the 3x3 matrix (row-wise):\n");
        for (int i = 0; i < ROWS * COLS; i++) {
            scanf("%d", &matrix[i]);
        }

        printf("Enter the element to search for: ");
        scanf("%d", &search_element);

        // Print the matrix for clarity
        printf("\nMatrix entered:\n");
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                printf("%d ", matrix[i * COLS + j]);
            }
            printf("\n");
        }
    }

    // Broadcast the search element to all processes
    MPI_Bcast(&search_element, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the matrix rows to all processes
    // Each process gets one row (3 elements)
    MPI_Scatter(matrix, COLS, MPI_INT, local_row, COLS, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process counts the occurrences of the search element in its row
    for (int i = 0; i < COLS; i++) {
        if (local_row[i] == search_element) {
            local_count++;
        }
    }

    // Reduce the counts from all processes to get the total count at the root
    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints the result
    if (rank == 0) {
        printf("Element %d appears %d times in the matrix.\n", search_element, total_count);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

// mpicc -o mpi_matrix_search mpi_matrix_search.c
// mpirun -np 3 ./mpi_matrix_search