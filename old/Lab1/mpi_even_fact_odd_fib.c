#include <mpi.h>
#include <stdio.h>

// Function to calculate factorial
unsigned long long factorial(int n) {
    if (n == 0 || n == 1)
        return 1;
    unsigned long long fact = 1;
    for (int i = 2; i <= n; i++) {
        fact *= i;
    }
    return fact;
}

// Function to calculate the nth Fibonacci number
unsigned long long fibonacci(int n) {
    if (n == 0)
        return 0;
    if (n == 1)
        return 1;
    unsigned long long a = 0, b = 1, c;
    for (int i = 2; i <= n; i++) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

int main(int argc, char** argv) {
    int rank, size;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check whether the rank is even or odd
    if (rank % 2 == 0) {
        // Even-ranked process: calculate factorial
        unsigned long long fact = factorial(rank);
        printf("Process %d (Even): Factorial(%d) = %llu\n", rank, rank, fact);
    } else {
        // Odd-ranked process: calculate Fibonacci
        unsigned long long fib = fibonacci(rank);
        printf("Process %d (Odd): Fibonacci(%d) = %llu\n", rank, rank, fib);
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}

//Process 2 (Even): Factorial(2) = 2
//Process 3 (Odd): Fibonacci(3) = 2
//Process 0 (Even): Factorial(0) = 1
//Process 1 (Odd): Fibonacci(1) = 1
