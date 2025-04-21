#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define MAX_LEN 100

// Function to toggle case
void toggle_case(char *str) {
    for (int i = 0; str[i] != '\0'; i++) {
        if (islower(str[i]))
            str[i] = toupper(str[i]);
        else if (isupper(str[i]))
            str[i] = tolower(str[i]);
    }
}

int main(int argc, char *argv[]) {
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char word[MAX_LEN];

    if (rank == 0) {
        // Process 0: Sending a word
        strcpy(word, "HelloMPI"); // Example word
        printf("Process 0 sending: %s\n", word);
        MPI_Ssend(word, MAX_LEN, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

        // Receiving toggled word
        MPI_Recv(word, MAX_LEN, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 0 received: %s\n", word);
    } 
    else if (rank == 1) {
        // Process 1: Receiving the word
        MPI_Recv(word, MAX_LEN, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        toggle_case(word);
        printf("Process 1 toggled and sending: %s\n", word);

        // Sending the toggled word back
        MPI_Ssend(word, MAX_LEN, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

// mpicc -o mpi_sync_send mpi_sync_send.c
// mpirun -np 2 ./mpi_sync_send
