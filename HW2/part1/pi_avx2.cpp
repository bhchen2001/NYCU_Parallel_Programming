# include <iostream>
# include <random>
#include <climits>
#include <ctime>
#include <pthread.h>
#include <sys/random.h>
#include <string.h>
#include <float.h>
#include "shishua-avx2.h"
using namespace std;

// shared variable for all thread
long long int number_in_circle = 0;
pthread_mutex_t num_circle_mutex;

void *pi_estimate(void *arg){
    long long int num_task = *(int*)arg;
    long long int number_in_circle_local = 0;

    prng_state prng;
    uint64_t seed[4];
    getrandom((void *)seed, sizeof(uint64_t) * 4, 0);
    prng_init(&prng, seed);
    // buf stores 128/8 = 16 random numbers
    // 8 for x and 8 for y
    uint8_t buf[128];

    for(long long int i = 0; i < num_task; i+=8){
        prng_gen(&prng, buf, sizeof(buf));
        // convert random number in buf into double
        for(int idx = 0; idx < 8; idx++){
            double tmpx = ((double)(*((uint32_t *)&buf[idx * 8])) / UINT32_MAX) * 2 - 1;
            double tmpy = ((double)(*((uint32_t *)&buf[(idx + 8) * 8])) / UINT32_MAX) * 2 - 1;
            // calculate the distance
            double distance_squared = tmpx * tmpx + tmpy * tmpy;
            if(distance_squared <= 1) number_in_circle_local++;
        }
    }

    pthread_mutex_lock(&num_circle_mutex);
    number_in_circle += number_in_circle_local;
    pthread_mutex_unlock(&num_circle_mutex);
    pthread_exit(NULL);
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
    long long int num_task[4];
    for(int i = 0; i < 4; i++){
        num_task[i] = number_of_toss/num_th;
    }
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