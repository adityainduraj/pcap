#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, value;

    MPI_Init(&argc, &argv);                     // Initialize MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);       // Get total number of processes

    // Ensure at least 2 processes
    if (size < 2) {
        if (rank == 0) {
            printf("Error: This program requires at least 2 processes.\n");
            fflush(stdout);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        // Root process initializes the value
        printf("Enter an integer value: ");
        fflush(stdout);  // Ensure prompt appears
        scanf("%d", &value);

        // Send the value to the first process
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Process %d sent value %d to Process 1\n", rank, value);
        fflush(stdout);

        // Receive final incremented value from last process
        MPI_Recv(&value, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Final value received back at root process: %d\n", value);
        fflush(stdout);
    } 
    else {
        // Receive value from previous process
        MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        value += 1;  // Increment received value

        if (rank == size - 1) {
            // Last process sends the value back to root
            MPI_Send(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            printf("Process %d (last) incremented value to %d and sent back to root\n", rank, value);
            fflush(stdout);
        } 
        else {
            // Intermediate process sends the incremented value to the next process
            MPI_Send(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            printf("Process %d incremented value to %d and sent to Process %d\n", rank, value, rank + 1);
            fflush(stdout);
        }
    }

    MPI_Finalize(); // Finalize MPI environment
    return 0;
}

