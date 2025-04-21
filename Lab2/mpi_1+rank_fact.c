#include <mpi.h>
#include <stdio.h>

// Function to compute factorial
long long factorial(int num) {
    if (num == 0 || num == 1) return 1;
    long long fact = 1;
    for (int i = 2; i <= num; i++) {
        fact *= i;
    }
    return fact;
}

int main(int argc, char *argv[]) {
    int rank, size, N, i, sum = 0, partial_sum = 0;
    long long local_factorial, global_sum = 0;
    fflush(stdout);
    MPI_Init(&argc, &argv);                   // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);     // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);     // Get total number of processes

    if (rank == 0) {
        // Root process reads the value of N
        printf("Enter value of N: ");
        fflush(stdout);
        scanf("%d", &N);
        
        // Broadcast N to all processes
        for (i = 1; i < size; i++) {
            MPI_Send(&N, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        // Receive N from root process
        MPI_Recv(&N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Compute sum of first 'rank' numbers
    for (i = 1; i <= rank; i++) {
        partial_sum += i;
    }

    // Compute factorial of the sum
    local_factorial = factorial(partial_sum);

    // Reduce all factorials to root process (sum them up)
    MPI_Reduce(&local_factorial, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Print final result at root process
        printf("Final result: %lld\n", global_sum);
    }

    MPI_Finalize();  // Finalize MPI
    return 0;
}

// mpicc -o mpi_factorial mpi_factorial.c
// mpirun -np 5 ./mpi_factorial
