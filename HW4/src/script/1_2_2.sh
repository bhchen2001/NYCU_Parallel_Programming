num_p=4

mpicxx ../pi_block_tree.cc -o ../pi_block_tree
./scp_script.sh
mpirun -np 16 --hostfile /home/.grade/HW4/4_hosts ../pi_block_tree 1000000000
mpirun -np 16 --hostfile /home/.grade/HW4/4_hosts ../../ref/pi_block_tree 1000000000