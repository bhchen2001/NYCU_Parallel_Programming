# include <iostream>
# include <random>
#include <climits>
#include <ctime>
using namespace std;

int main(){
    srand(time(NULL));
    long long int number_of_toss = 1000000000;
    cout << "number of toss: " << number_of_toss << endl;
    long long int number_in_circle = 0;
    for(long long int i = 0; i < number_of_toss; i++){
        double x, y;
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        x = -1 + x * 2;
        y = -1 + y * 2;

        // much slower way to generate random number
        // random_device rd_real;
        // mt19937 gen_real(rd_real());
        // uniform_real_distribution<> real_random_generator(-1, 1);
        // double x = real_random_generator(gen_real);
        // double y = real_random_generator(gen_real);
        
        double distance_squared = x * x + y * y;
        if(distance_squared <= 1) number_in_circle++;
    }
    double pi_estimate = 4 * number_in_circle / ((double)number_of_toss);
    cout << "pi_estimate: " << pi_estimate << endl;
}