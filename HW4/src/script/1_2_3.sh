num_p=4

mpicxx ../pi_nonblock_linear.cc -o ../pi_nonblock_linear
./scp_script.sh
mpirun -np $num_p --hostfile hosts ../pi_nonblock_linear 1000000000
mpirun -np $num_p --hostfile hosts ../../ref/pi_nonblock_linear 1000000000