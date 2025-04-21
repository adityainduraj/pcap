#include <stdio.h>
#include <mpi.h>
#define mcw MPI_COMM_WORLD

int main(int argc, char **argv) {
    int rank, size;
    double n1, n2, res;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(mcw, &rank);
    MPI_Comm_size(mcw, &size);
    if (size<4) {
        if (rank == 0) {
            printf("This program requires 4 processes\n");
        }
        MPI_Finalize();
        return 0;
    }
    if (rank==0) {
        printf("Enter two numbers: ");
        scanf("%lf %lf", &n1, &n2);
    }
    MPI_Barrier(mcw);
    MPI_Bcast(&n1, 1, MPI_DOUBLE, 0, mcw);
    MPI_Bcast(&n2, 1, MPI_DOUBLE, 0, mcw);
    switch(rank) {
        case 0:
            res=n1+n2;
            printf("Process %d : %lf + %lf = %lf\n", rank, n1, n2, res);
            break;
        case 1:
            res=n1-n2;
            printf("Process %d : %lf - %lf = %lf\n", rank, n1, n2, res);
            break;
        case 2:
            res=n1*n2;
            printf("Process %d : %lf * %lf = %lf\n", rank, n1, n2, res);
            break;
        case 3:
            if (n2!=0){
                res=n1/n2;
                printf("Process %d : %lf / %lf = %lf\n", rank, n1, n2, res);
            }
            else {
            printf("Division is invalid\n");
            }
            break;
        default:
            printf("Enter a valid operator\n");
            break;
    }
    MPI_Finalize();
    return 0;
}