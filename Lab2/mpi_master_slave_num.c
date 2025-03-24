#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int number;

    MPI_Init(&argc, &argv);                     // Initialize the MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size);       // Get the total number of processes

    if (rank == 0) {
        // Master process
        for (int i = 1; i < size; i++) {
            number = i * 100; // Example: send different number to each slave
            printf("Master (process 0) sending %d to process %d\n", number, i);
            MPI_Send(&number, 1, MPI_INT, i, 0, MPI_COMM_WORLD); // Standard send
        }
    } else {
        // Slave processes
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Slave process %d received number: %d\n", rank, number);
    }

    MPI_Finalize(); // Finalize the MPI environment
    return 0;
}

// mpicc -o mpi_send_program mpi_send_program.c
// mpirun -np 4 ./mpi_send_program
