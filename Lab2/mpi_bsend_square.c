#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int number;
    int *arr = NULL;
    int buffer_size;
    void *buffer;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate buffer for MPI_Bsend
    buffer_size = size * (sizeof(int) + MPI_BSEND_OVERHEAD);
    buffer = malloc(buffer_size);
    MPI_Buffer_attach(buffer, buffer_size);

    if (rank == 0) {
        // Root process (process 0): Generate an array and send elements
        arr = (int *)malloc(size * sizeof(int));
        printf("Root process initializing array:\n");
        for (int i = 0; i < size; i++) {
            arr[i] = i + 1; // Assign some values
            printf("%d ", arr[i]);
        }
        printf("\n");

        // Send one value to each process (including itself for uniformity)
        for (int i = 1; i < size; i++) {
            MPI_Bsend(&arr[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        free(arr);
    } else {
        // Slave processes: Receive data and compute result
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int result;
        if (rank % 2 == 0) {
            result = number * number; // Even process: Square
            printf("Process %d received %d, squared result: %d\n", rank, number, result);
        } else {
            result = number * number * number; // Odd process: Cube
            printf("Process %d received %d, cubed result: %d\n", rank, number, result);
        }
    }

    // Detach and free buffer
    MPI_Buffer_detach(&buffer, &buffer_size);
    free(buffer);

    MPI_Finalize();
    return 0;
}
