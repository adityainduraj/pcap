#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size, M, N, total_elements;
    double local_avg = 0.0, global_avg = 0.0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    N = size; // Number of processes
    int *array = NULL; 
    int *local_array = (int *)malloc(M * sizeof(int));
    
    if (rank == 0) {
        printf("Enter the number of elements per process (M): ");
        scanf("%d", &M);

        total_elements = N * M;
        array = (int *)malloc(total_elements * sizeof(int));

        printf("Enter %d elements: ", total_elements);
        for (int i = 0; i < total_elements; i++) {
            scanf("%d", &array[i]);
        }
    }

    // Broadcast M to all processes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter M elements to each process
    MPI_Scatter(array, M, MPI_INT, local_array, M, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local average
    int sum = 0;
    for (int i = 0; i < M; i++) {
        sum += local_array[i];
    }
    local_avg = (double)sum / M;

    // Gather all averages at root process
    MPI_Reduce(&local_avg, &global_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Compute final average at root process
    if (rank == 0) {
        global_avg = global_avg / N;
        printf("Final average: %.2f\n", global_avg);
        free(array);
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}

// mpicc -o mpi_average mpi_average.c
// mpirun -np 4 ./mpi_average
