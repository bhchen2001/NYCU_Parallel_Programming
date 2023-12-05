# compile the 1.2 requirement and run it
mpicxx ../pi_block_linear.cc -o ../pi_block_linear
./scp_script.sh
mpirun -np 4 --hostfile hosts ../pi_block_linear 1000000000
mpirun -np 4 --hostfile hosts ../../ref/pi_block_linear 1000000000