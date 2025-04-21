#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check the rank and print the appropriate message
    if (rank % 2 == 0) {
        printf("Process %d: Hello\n", rank);
    } else {
        printf("Process %d: World\n", rank);
    }

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}

//Process 1: World
//Process 2: Hello
//Process 0: Hello
