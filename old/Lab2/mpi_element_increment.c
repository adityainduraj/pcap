#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, value;

    MPI_Init(&argc, &argv);                     // Initialize MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);       // Get total number of processes

    if (rank == 0) {
        // Root process initializes the value
        printf("Enter an integer value: ");
        scanf("%d", &value);

        // Send the value to the first process
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Process %d sent value %d to Process 1\n", rank, value);

        // Receive final incremented value from last process
        MPI_Recv(&value, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Final value received back at root process: %d\n", value);
    } 
    else {
        // Receive value from previous process
        MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        value += 1;  // Increment received value

        if (rank == size - 1) {
            // Last process sends the value back to root
            MPI_Send(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            printf("Process %d (last) incremented value to %d and sent back to root\n", rank, value);
        } 
        else {
            // Intermediate process sends the incremented value to the next process
            MPI_Send(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            printf("Process %d incremented value to %d and sent to Process %d\n", rank, value, rank + 1);
        }
    }

    MPI_Finalize(); // Finalize MPI environment
    return 0;
}

// mpicc -o mpi_chain mpi_chain.c
// mpirun -np 4 ./mpi_chain
