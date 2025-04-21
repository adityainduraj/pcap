#include <mpi.h>
#include <stdio.h>
#include <math.h>

// Function to check if a number is prime
int is_prime(int num) {
    if (num < 2) return 0;
    for (int i = 2; i <= sqrt(num); i++) {
        if (num % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int rank, size, num;

    MPI_Init(&argc, &argv);                     // Initialize MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);       // Get total number of processes

    int arr[size];  // Array to hold elements, size should match number of processes

    if (rank == 0) {
        // Master process reads the array
        printf("Enter %d elements: ", size);
        for (int i = 0; i < size; i++) {
            scanf("%d", &arr[i]);
        }

        // Send each process its corresponding element
        for (int i = 1; i < size; i++) {
            MPI_Send(&arr[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        num = arr[0];  // Master process keeps the first element
    } else {
        // Receive the number assigned to this process
        MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Each process checks if its number is prime
    int result = is_prime(num);

    // Print the result from each process
    printf("Process %d received %d, Prime: %s\n", rank, num, result ? "YES" : "NO");

    MPI_Finalize(); // Finalize MPI environment
    return 0;
}

// mpicc -o mpi_prime mpi_prime.c -lm
// mpirun -np 5 ./mpi_prime