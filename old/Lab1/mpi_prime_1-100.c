#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Function to check if a number is prime
bool is_prime(int num) {
    if (num < 2) return false;
    for (int i = 2; i * i <= num; i++) {
        if (num % i == 0) return false;
    }
    return true;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int start, end;
    int local_primes[50]; // Local storage for found primes
    int local_count = 0;
    int total_primes[100]; // To store all prime numbers (final result)
    int total_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            printf("This program requires exactly 2 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Divide the range between two processes
    start = (rank == 0) ? 1 : 51;
    end = (rank == 0) ? 50 : 100;

    // Find prime numbers in the assigned range
    for (int i = start; i <= end; i++) {
        if (is_prime(i)) {
            local_primes[local_count++] = i;
        }
    }

    // Gather results in process 0
    MPI_Gather(&local_count, 1, MPI_INT, &total_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_primes, local_count, MPI_INT, total_primes, local_count, MPI_INT, 0, MPI_COMM_WORLD);

    // Process 0 prints the results
    if (rank == 0) {
        printf("Prime numbers between 1 and 100: ");
        for (int i = 0; i < total_count; i++) {
            printf("%d ", total_primes[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}

// mpicc -o mpi_primes mpi_primes.c
// mpirun -np 2 ./mpi_primes
