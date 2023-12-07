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

    // TODO: binary tree redunction
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

    if(world_rank % 2 == 1){
        MPI_Send(&number_in_circle_local, 1, MPI_LONG_LONG_INT, world_rank - 1, 0, MPI_COMM_WORLD);
    }
    else{
        for(int factor = 1; factor < world_size; factor *= 2){
            int flag = world_rank / factor;
            if(flag % 2 == 0){
                long long int number_in_circle_rcv = 0;
                MPI_Recv(&number_in_circle_rcv, 1, MPI_LONG_LONG_INT, world_rank + factor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                number_in_circle_local += number_in_circle_rcv;
            }
            else{
                MPI_Send(&number_in_circle_local, 1, MPI_LONG_LONG_INT, world_rank - factor, 0, MPI_COMM_WORLD);
                break;
            }
        }
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4 * number_in_circle_local / ((double)tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
