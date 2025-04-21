#include <mpi.h>
#include <stdio.h>

// Function to compute factorial
long long factorial(int n) {
    long long fact = 1;
    for (int i = 2; i <= n; i++) {
        fact *= i;
    }
    return fact;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int N;
    int value;
    long long fact, total_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process reads N values
        printf("Enter the number of values (N): ");
        scanf("%d", &N);

        if (N != size) {
            printf("Run with exactly N processes!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int values[N];
        printf("Enter %d values: ", N);
        for (int i = 0; i < N; i++) {
            scanf("%d", &values[i]);
        }

        // Send one value to each worker process
        for (int i = 0; i < N; i++) {
            MPI_Send(&values[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }

    // All processes receive a value, compute factorial, and return result
    MPI_Recv(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    fact = factorial(value);

    // Root gathers results
    MPI_Reduce(&fact, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints the final result
    if (rank == 0) {
        printf("Sum of factorials: %lld\n", total_sum);
    }

    MPI_Finalize();
    return 0;
}

// mpicc -o mpi_fact_sum mpi_fact_sum.c
// mpirun -np 4 ./mpi_fact_sum
