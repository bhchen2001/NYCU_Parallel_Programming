num_p=4

mpicxx ../pi_one_side.cc -o ../pi_one_side
./scp_script.sh
mpirun -np $num_p --hostfile hosts ../pi_one_side 1000000000
mpirun -np $num_p --hostfile hosts ../../ref/pi_one_side 1000000000