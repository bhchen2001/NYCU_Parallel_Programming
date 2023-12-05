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

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    long long int number_in_circle = 0;
    long long int number_in_circle_local = 0;
    long long int local_tosses = tosses / world_size;
    unsigned int seed = time(0) * world_rank;

    for(long long int i = 0; i < local_tosses; i++){
        double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1){
            number_in_circle_local++;
        }
    }

    MPI_Win_create(&number_in_circle, sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    if (world_rank == 0)
    {
        // Master
        number_in_circle += number_in_circle_local;
    }
    else
    {
        // Workers
        // lock the window
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Accumulate(&number_in_circle_local, 1, MPI_LONG_LONG_INT, 0, 0, 1, MPI_LONG_LONG_INT, MPI_SUM, win);
        MPI_Win_unlock(0, win);
    }

    MPI_Win_fence(0, win);
    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
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