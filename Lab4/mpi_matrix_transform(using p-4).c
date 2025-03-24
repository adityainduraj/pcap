#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ROWS 4
#define COLS 4
#define OUTPUT_ROWS 5

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[ROWS * COLS];          // Input matrix stored as 1D array
    int local_row[COLS];              // Each process gets one row
    int local_output[COLS];           // Each process's output row
    int output_matrix[OUTPUT_ROWS * COLS]; // Final output matrix at root

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if the number of processes is exactly 4
    if (size != 4) {
        if (rank == 0) {
            fprintf(stderr, "Error: This program requires exactly 4 processes, but %d were provided.\n", size);
        }
        MPI_Finalize();
        return 1;
    }

    // Root process reads the 4x4 matrix
    if (rank == 0) {
        printf("Enter the elements of the 4x4 matrix (row-wise):\n");
        for (int i = 0; i < ROWS * COLS; i++) {
            scanf("%d", &matrix[i]);
        }

        // Print the input matrix for clarity
        printf("\nInput Matrix:\n");
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                printf("%d ", matrix[i * COLS + j]);
            }
            printf("\n");
        }
    }

    // Scatter the matrix rows to all processes
    MPI_Scatter(matrix, COLS, MPI_INT, local_row, COLS, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process computes its output row based on the rank
    if (rank == 0) {
        // Process 0: First row of output is the same as the first row of input
        for (int j = 0; j < COLS; j++) {
            local_output[j] = local_row[j];
        }
    } else if (rank == 1) {
        // Process 1: Multiply first row by 2, adjust the last element
        for (int j = 0; j < COLS - 1; j++) {
            local_output[j] = local_row[0] * 2;  // local_row[0] is first row
        }
        local_output[COLS - 1] = local_row[0] + 4;  // Adjust last element (e.g., 1 -> 5)
    } else if (rank == 2) {
        // Process 2: Same as Process 1's output
        for (int j = 0; j < COLS - 1; j++) {
            local_output[j] = local_row[0] * 2;  // local_row[0] is first row
        }
        local_output[COLS - 1] = local_row[0] + 4;
    } else if (rank == 3) {
        // Process 3: Increment Process 1's output by 1 (with adjustment)
        for (int j = 0; j < COLS - 1; j++) {
            local_output[j] = (local_row[0] * 2) + 1;  // local_row[0] is first row
        }
        local_output[COLS - 1] = local_row[0] + 4 + 1;
    }

    // Gather the output rows from all processes to the root
    MPI_Gather(local_output, COLS, MPI_INT, output_matrix, COLS, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process computes the fifth row and prints the output matrix
    if (rank == 0) {
        // Compute the fifth row based on the fourth row (Process 3's output)
        for (int j = 0; j < COLS - 1; j++) {
            output_matrix[4 * COLS + j] = output_matrix[3 * COLS + j] + 2;
        }
        output_matrix[4 * COLS + (COLS - 1)] = output_matrix[3 * COLS + (COLS - 1)] + 1;

        // Print the output matrix
        printf("\nOutput Matrix:\n");
        for (int i = 0; i < OUTPUT_ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                printf("%d ", output_matrix[i * COLS + j]);
            }
            printf("\n");
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

// mpicc -o mpi_matrix_transform mpi_matrix_transform.c
// mpirun -np 4 ./mpi_matrix_transform