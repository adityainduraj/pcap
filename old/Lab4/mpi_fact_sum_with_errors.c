#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to compute factorial of a number
unsigned long long factorial(int n) {
    if (n <= 1) return 1;
    unsigned long long result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int err_code;
    unsigned long long local_fact, partial_sum;

    // Initialize MPI with error handling
    err_code = MPI_Init(&argc, &argv);
    if (err_code != MPI_SUCCESS) {
        char error_string[BUFSIZ];
        int length;
        MPI_Error_string(err_code, error_string, &length);
        fprintf(stderr, "MPI_Init failed: %s\n", error_string);
        return 1;
    }

    // Get rank and size with error handling
    err_code = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (err_code != MPI_SUCCESS) {
        char error_string[BUFSIZ];
        int length;
        MPI_Error_string(err_code, error_string, &length);
        fprintf(stderr, "MPI_Comm_rank failed: %s\n", error_string);
        MPI_Finalize();
        return 1;
    }

    err_code = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (err_code != MPI_SUCCESS) {
        char error_string[BUFSIZ];
        int length;
        MPI_Error_string(err_code, error_string, &length);
        fprintf(stderr, "MPI_Comm_size failed: %s\n", error_string);
        MPI_Finalize();
        return 1;
    }

    // Check if size is at least 1
    if (size < 1) {
        if (rank == 0) {
            fprintf(stderr, "Error: Number of processes must be at least 1, got %d\n", size);
        }
        MPI_Finalize();
        return 1;
    }

    // Each process computes factorial based on its rank
    // Rank 0 computes 1!, Rank 1 computes 2!, ..., Rank (size-1) computes size!
    local_fact = factorial(rank + 1);

    // Use MPI_Scan to compute the partial sum up to each process
    err_code = MPI_Scan(&local_fact, &partial_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (err_code != MPI_SUCCESS) {
        char error_string[BUFSIZ];
        int length;
        MPI_Error_string(err_code, error_string, &length);
        fprintf(stderr, "MPI_Scan failed at rank %d: %s\n", rank, error_string);
        MPI_Finalize();
        return 1;
    }

    // Process with rank (size-1) computes (size+1)! and adds it to the sum
    unsigned long long final_sum = partial_sum;
    if (rank == size - 1) {
        unsigned long long next_fact = factorial(size + 1);
        final_sum += next_fact;
        printf("Sum of factorials from 1! to %d! = %llu\n", size + 1, final_sum);
    }

    // Finalize MPI
    err_code = MPI_Finalize();
    if (err_code != MPI_SUCCESS) {
        char error_string[BUFSIZ];
        int length;
        MPI_Error_string(err_code, error_string, &length);
        fprintf(stderr, "MPI_Finalize failed: %s\n", error_string);
        return 1;
    }

    return 0;
}

// mpicc -o mpi_factorial_sum mpi_factorial_sum.c
// mpirun -np 3 ./mpi_factorial_sum