num_p=4

mpicxx pi_block_tree.cc -o pi_block_tree
./scp_script.sh
mpirun -np $num_p --hostfile hosts pi_block_tree 1000000000
mpirun -np $num_p --hostfile hosts ../ref/pi_block_tree 1000000000