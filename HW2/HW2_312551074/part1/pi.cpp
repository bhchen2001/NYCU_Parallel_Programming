# include <iostream>
# include <random>
#include <climits>
#include <ctime>
#include <pthread.h>
using namespace std;

// shared variable for all thread
long long int number_in_circle = 0;
pthread_mutex_t num_circle_mutex;

void *pi_estimate(void *arg){
    long long int num_task = *(int*)arg;
    long long int number_in_circle_local = 0;
    // thread safe random number generator
    // unsigned int seed = time(1);
    unsigned int seed = 777;
    for(long long int i = 0; i < num_task; i++){
        double x, y;
        x = (double)rand_r(&seed) / RAND_MAX;
        y = (double)rand_r(&seed) / RAND_MAX;
        x = -1 + x * 2;
        y = -1 + y * 2;

        double distance_squared = x * x + y * y;
        if(distance_squared <= 1) number_in_circle_local++;
    }
    pthread_mutex_lock(&num_circle_mutex);
    number_in_circle += number_in_circle_local;
    pthread_mutex_unlock(&num_circle_mutex);
}

int main(int argc, char* argv[]){

    if(argc != 3){
        perror("invalid argment");
        exit(1);
    }

    int num_th = atoi(argv[1]);
    long long int number_of_toss = atoll(argv[2]);
    // cout << "number of thread:" << num_th << "\nnumber of toss: " << number_of_toss << endl;

    // the number of task that each thread will be assigned
    long long int num_task = number_of_toss/num_th;
    // allocate the thread pool according to the user-defined thread number
    pthread_t *thread_pool = (pthread_t*)malloc(num_th * sizeof(pthread_t));
    pthread_mutex_init(&num_circle_mutex, NULL);
    
    for(int i = 0; i < num_th; i++){
        pthread_create((thread_pool + i), NULL, *pi_estimate, (void*)&num_task);
    }

    for(int i = 0; i < num_th; i++){
        pthread_join(thread_pool[i], NULL);
    }

    free(thread_pool);
    pthread_mutex_destroy(&num_circle_mutex);

    double pi_estimate = 4 * number_in_circle / ((double)number_of_toss);
    cout << pi_estimate << endl;
}