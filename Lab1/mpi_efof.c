#include <mpi.h>
#include <stdio.h>
#define mcw MPI_COMM_WORLD
unsigned long long factorial(int n) {
    if (n==1||n==1)
        return 1;
    unsigned long long fact = 1;
    for (inti=2; i<=n; i++)
        fact *= i;
    return fact;
}
unisgned long long fibonacci(int n) {
    if (n==0)
        return 0;
    if (n==1)
        return 1;
    unsigned long long a=0, b=1, c;
    for (int i=2; i<=n; i++) {
        c=a+b;
        a=b;
        b=c;
    }
    return b;
}

int main(int argc, char*argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(mcw, &rank);
    MPI_Comm_size(mcw, &size);
    if (rank%2==0) {
        unsigned long long fact = factorial(rank);
        printf("Process %d has factorial: %llu", rank, fact);
    } else if (rank%2!==0) {
        unsigned long long fib = fibonacci(rank);
        printf("Process %d has factorial: %llu", rank, fib);
    }
    MPI_Finalize();
    return 0;
}