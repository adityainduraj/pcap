// MPI_Gather – sending numbers
#include <mpi.h>  
#include <stdio.h>
 #include <stdlib.h>
 int main (int argc, char *argv[]) {
    int root_rank = 0, size, my_rank;
    int send_buf = 111;
    int recv_buf[4];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size != 4)
    {
        printf("This application is meant to be run with 4 MPI processes.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
     MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
     send_buf = send_buf + my_rank;
      
    // Each MPI process sends its send_buf to Root proess which collects the data in rank order
    printf("Sending data %d from from process %d \n", send_buf, my_rank);
    MPI_Gather(&send_buf, 1, MPI_INT, &recv_buf, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
    
    // Display result in Root
     if(my_rank == root_rank)
    {
    printf("The gathered data at Root is \n");
    for(int i=0; i<4; i++)
        printf("%d \n", recv_buf[i]);
    }
 
    MPI_Finalize();
    return 0;

// MPI_Gather – sending Strings 
#include <mpi.h>  
#include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 int main (int argc, char *argv[]) {
    int root_rank = 0, size, my_rank;
    char send_buf[10] = "AA ";
    char *recv_buf;
              
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    int len = strlen(send_buf);
    send_buf[len-2] = send_buf[len-2] + my_rank;
    recv_buf = (char *)malloc(size*len*sizeof(char)); 
    
    // Each MPI process sends its send_buf to Root proess which collects the data in rank order
    printf("Sending data %s from from process %d \n", send_buf, my_rank);
    MPI_Gather(send_buf, len, MPI_CHAR, recv_buf, len, MPI_CHAR, root_rank, MPI_COMM_WORLD);
    
    // Display result in Root
     if(my_rank == root_rank)
       printf("The gathered data at Root is : %s \n", recv_buf);
 
    MPI_Finalize();
    return 0;
 }


//  All Gather
#include <mpi.h>  
#include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 int main (int argc, char *argv[]) {
    int size, my_rank;
    char send_buf[10] = "AA ";
    char *recv_buf;
              
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    int len = strlen(send_buf);
    send_buf[len-2] = send_buf[len-2] + my_rank;
    recv_buf = (char *)malloc(size*len*sizeof(char)); 
    
    // Each MPI process sends its send_buf to Root proess which collects the data in rank order
    printf("Sending data %s from from process %d \n", send_buf, my_rank);
    MPI_Allgather(send_buf, len, MPI_CHAR, recv_buf, len, MPI_CHAR, MPI_COMM_WORLD);
    
    // Display gathered result 
    printf("The gathered data at Process [%d] is : %s \n", my_rank, recv_buf);
 
    MPI_Finalize();
    return 0;
 }
