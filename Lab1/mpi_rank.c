#include <mpi.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char** argv) {
    int rank, size, x;
    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // The root process takes input
        printf("Enter the value of x (integer): ");
        fflush(stdout);  // Ensure prompt is displayed
        scanf("%d", &x);
    }

    // Broadcast the value of x to all processes
    MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process calculates pow(x, rank)
    double result = pow(x, rank);

    // Print the result from each process
    printf("Process %d: pow(%d, %d) = %f\n", rank, x, rank, result);

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}

//Enter the value of x (integer): 5    
//Process 1: pow(5, 1) = 5.000000
//Process 2: pow(5, 2) = 25.000000
//Process 0: pow(5, 0) = 1.000000
