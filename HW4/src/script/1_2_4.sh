num_p=4

mpicxx ../pi_gather.cc -o ../pi_gather
./scp_script.sh
mpirun -np $num_p --hostfile hosts ../pi_gather 1000000000
mpirun -np $num_p --hostfile hosts ../../ref/pi_gather 1000000000