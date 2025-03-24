#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

int main(int argc, char** argv) {
    int rank, size;
    char str[100];

    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process takes the input string
        printf("Enter a string: ");
        fflush(stdout); // Ensure prompt is displayed
        scanf("%s", str);
    }

    // Broadcast the string to all processes
    MPI_Bcast(str, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Get the length of the string
    int len = strlen(str);

    if (rank < len) {
        // Toggle the character at the index equal to the process rank
        if (isupper(str[rank])) {
            str[rank] = tolower(str[rank]);
        } else if (islower(str[rank])) {
            str[rank] = toupper(str[rank]);
        }
    }

    // Gather the modified strings back to the root process
    char result[100];
    MPI_Gather(&str[rank], 1, MPI_CHAR, result, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Root process prints the final toggled string
    if (rank == 0) {
        printf("Toggled string: %s\n", result);
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}

//Enter a string: DOG
//Toggled string: dog
