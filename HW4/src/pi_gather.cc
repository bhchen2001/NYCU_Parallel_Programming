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
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    long long int number_in_circle = 0;
    long long int number_in_circle_local = 0;
    long long int local_tosses = tosses / world_size;
    long long int *number_in_circle_rcv = new long long int[world_size];
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

    // TODO: use MPI_Gather
    MPI_Gather(&number_in_circle_local, 1, MPI_LONG_LONG_INT, number_in_circle_rcv, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // TODO: PI result
        for (int i = 0; i < world_size; i++){
            number_in_circle += number_in_circle_rcv[i];
        }
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
