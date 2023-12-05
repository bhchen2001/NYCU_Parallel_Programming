num_p=4

mpicxx ../pi_reduce.cc -o ../pi_reduce
./scp_script.sh
mpirun -np $num_p --hostfile hosts ../pi_reduce 1000000000
mpirun -np $num_p --hostfile hosts ../../ref/pi_reduce 1000000000