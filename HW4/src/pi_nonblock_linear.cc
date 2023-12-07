#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // monte carlo
    long long int number_in_circle = 0;
    long long int number_in_circle_local = 0;
    long long int local_tosses = tosses / world_size;
    unsigned int seed =  time(0) * (world_rank + 1);

    for(long long int i = 0; i < local_tosses; i++){
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;
        x = -1 + x * 2;
        y = -1 + y * 2;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1){
            number_in_circle_local++;
        }
    }


    if (world_rank > 0)
    {
        // TODO: MPI workers
        MPI_Request request;
        MPI_Isend(&number_in_circle_local, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        number_in_circle += number_in_circle_local;
        MPI_Request *requests = new MPI_Request[world_size - 1];
        long long int *number_in_circle_rcv = new long long int[world_size - 1];
        for (int i = 1; i < world_size; i++){
            MPI_Irecv(&number_in_circle_rcv[i - 1], 1, MPI_LONG_LONG_INT, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
        }
        MPI_Waitall(world_size - 1, requests, MPI_STATUS_IGNORE);
        for (int i = 0; i < world_size - 1; i++){
            number_in_circle += number_in_circle_rcv[i];
        }
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4 * number_in_circle / ((double)tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
