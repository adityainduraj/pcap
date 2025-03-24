#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    double num1, num2, result;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 4) {
        if (rank == 0) {
            printf("This program requires at least 4 processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        // The root process takes two input numbers
        printf("Enter two numbers: ");
        scanf("%lf %lf", &num1, &num2);
    }

    // Broadcast the input numbers to all processes
    MPI_Bcast(&num1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform operations based on the process rank
    switch (rank) {
        case 0: // Addition
            result = num1 + num2;
            printf("Process %d: Addition (%.2lf + %.2lf) = %.2lf\n", rank, num1, num2, result);
            break;
        case 1: // Subtraction
            result = num1 - num2;
            printf("Process %d: Subtraction (%.2lf - %.2lf) = %.2lf\n", rank, num1, num2, result);
            break;
        case 2: // Multiplication
            result = num1 * num2;
            printf("Process %d: Multiplication (%.2lf * %.2lf) = %.2lf\n", rank, num1, num2, result);
            break;
        case 3: // Division
            if (num2 != 0) {
                result = num1 / num2;
                printf("Process %d: Division (%.2lf / %.2lf) = %.2lf\n", rank, num1, num2, result);
            } else {
                printf("Process %d: Division by zero is not allowed.\n", rank);
            }
            break;
        default:
            // Extra processes (if any) do nothing
            printf("Process %d: No operation assigned.\n", rank);
            break;
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}

//mpirun -np 4 ./l1q3
//6 7
//Enter two numbers: Process 0: Addition (6.00 + 7.00) = 13.00
//Process 2: Multiplication (6.00 * 7.00) = 42.00
//Process 1: Subtraction (6.00 - 7.00) = -1.00
//Process 3: Division (6.00 / 7.00) = 0.86
//
