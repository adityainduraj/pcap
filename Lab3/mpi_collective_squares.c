#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int M, N, total_elements;
    int *global_array = NULL;  // Array to store all elements at root
    int *local_array = NULL;   // Array for each process
    double *local_results = NULL;  // Array to store results in each process
    double *global_results = NULL; // Array to store all results at root

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process reads M and the array
    if (rank == 0) {
        printf("Enter the value of M: ");
        scanf("%d", &M);
        
        N = size;  // Number of processes
        total_elements = N * M;

        // Allocate memory for global array
        global_array = (int *)malloc(total_elements * sizeof(int));
        printf("Enter %d elements:\n", total_elements);
        for (int i = 0; i < total_elements; i++) {
            scanf("%d", &global_array[i]);
        }
    }

    // Broadcast M to all processes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Update N in all processes
    N = size;
    total_elements = N * M;

    // Allocate memory for local array in each process
    local_array = (int *)malloc(M * sizeof(int));
    local_results = (double *)malloc(M * sizeof(double));

    // Scatter the array to all processes
    MPI_Scatter(global_array, M, MPI_INT, local_array, M, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process computes based on its rank
    // Rank 0: square (pow 2)
    // Rank 1: cube (pow 3)
    // Rank 2: 4th power (pow 4), and so on
    for (int i = 0; i < M; i++) {
        local_results[i] = pow(local_array[i], rank + 2);
    }

    // Allocate memory for global results at root
    if (rank == 0) {
        global_results = (double *)malloc(total_elements * sizeof(double));
    }

    // Gather all results back to root
    MPI_Gather(local_results, M, MPI_DOUBLE, global_results, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Root process prints the results
    if (rank == 0) {
        printf("\nResults:\n");
        for (int i = 0; i < N; i++) {
            printf("Process %d (power %d):\n", i, i + 2);
            for (int j = 0; j < M; j++) {
                printf("%.2f ", global_results[i * M + j]);
            }
            printf("\n");
        }
    }

    // Free allocated memory
    if (rank == 0) {
        free(global_array);
        free(global_results);
    }
    free(local_array);
    free(local_results);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

// mpicc -o mpi_collective_squares mpi_collective_squares.c -lm
// mpirun -np 3 ./mpi_collective_squares
