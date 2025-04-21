#include <mpi.h>
#include <stdio.h>

#define N 4  // Matrix size (4x4)

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[N][N];
    int result[N][N];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        if (rank == 0) {
            printf("Please run with 4 processes.\n");
        }
        MPI_Finalize();
        return 0;
    }
    if (rank == 0) {
        printf("Enter a 4x4 matrix:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
    }
    MPI_Bcast(matrix, N * N, MPI_INT, 0, MPI_COMM_WORLD);
    for (int j = 0; j < N; j++) {
        result[rank][j] = 0;
        for (int i = 0; i <= rank; i++) {
            result[rank][j] += matrix[i][j];
        }
    }
    MPI_Gather(result[rank], N, MPI_INT, result, N, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Output matrix:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", result[i][j]);
            }
            printf("\n");
        }
    }
    MPI_Finalize();
    return 0;
}

// mpicc -o mpi_matrix_transform mpi_matrix_transform.c
// mpirun -np 4 ./mpi_matrix_transform
