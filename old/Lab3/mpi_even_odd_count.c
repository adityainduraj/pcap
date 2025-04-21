#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, N = 9;
    int A[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};  // Input array
    int local_size, local_A[9], local_result[9];
    int even_count = 0, odd_count = 0, local_even = 0, local_odd = 0;

    MPI_Init(&argc, &argv);                 // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // Get total number of processes

    local_size = N / size;  // Each process gets equal chunk

    // Scatter array elements to all processes
    MPI_Scatter(A, local_size, MPI_INT, local_A, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process processes its part
    for (int i = 0; i < local_size; i++) {
        if (local_A[i] % 2 == 0) {
            local_result[i] = 1;
            local_even++;
        } else {
            local_result[i] = 0;
            local_odd++;
        }
    }

    // Gather results back to root process
    MPI_Gather(local_result, local_size, MPI_INT, A, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Reduce even and odd counts to root
    MPI_Reduce(&local_even, &even_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_odd, &odd_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints the result
    if (rank == 0) {
        printf("Resultant Array (A): ");
        for (int i = 0; i < N; i++) {
            printf("%d ", A[i]);
        }
        printf("\nEven (Count)  = %d\n", even_count);
        printf("Odd  (Count)  = %d\n", odd_count);
    }

    MPI_Finalize();  // Finalize MPI
    return 0;
}

// mpicc -o mpi_array mpi_array.c
// mpirun -np 3 ./mpi_array
