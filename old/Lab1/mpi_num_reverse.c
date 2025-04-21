#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Function to reverse the digits of an integer
int reverse_number(int num) {
    int reversed = 0;
    while (num > 0) {
        reversed = reversed * 10 + num % 10;
        num /= 10;
    }
    return reversed;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int input_array[9] = {18, 523, 301, 1234, 2, 14, 108, 150, 1928};
    int output_array[9];
    int local_value, reversed_value;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 9) {
        if (rank == 0) {
            printf("This program requires exactly 9 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Each process gets one number
    local_value = input_array[rank];
    reversed_value = reverse_number(local_value);

    // Gather results back to process 0
    MPI_Gather(&reversed_value, 1, MPI_INT, output_array, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Reversed array: ");
        for (int i = 0; i < 9; i++) {
            printf("%d ", output_array[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}

// mpicc -o reverse_mpi reverse_mpi.c
// mpirun -np 9 ./reverse_mpi
