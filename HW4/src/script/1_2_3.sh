num_p=4

mpicxx ../pi_nonblock_linear.cc -o ../pi_nonblock_linear
./scp_script.sh
mpirun -np 5 --hostfile /home/.grade/HW4/2_hosts ../pi_nonblock_linear 1000000000
mpirun -np 5 --hostfile /home/.grade/HW4/2_hosts ../../ref/pi_nonblock_linear 1000000000
mpirun -np 7 --hostfile /home/.grade/HW4/2_hosts ../pi_nonblock_linear 1000000000
mpirun -np 7 --hostfile /home/.grade/HW4/2_hosts ../../ref/pi_nonblock_linear 1000000000
mpirun -np 10 --hostfile /home/.grade/HW4/3_hosts ../pi_nonblock_linear 1000000000
mpirun -np 10 --hostfile /home/.grade/HW4/3_hosts ../../ref/pi_nonblock_linear 1000000000
mpirun -np 11 --hostfile /home/.grade/HW4/3_hosts ../pi_nonblock_linear 1000000000
mpirun -np 11 --hostfile /home/.grade/HW4/3_hosts ../../ref/pi_nonblock_linear 1000000000
mpirun -np 16 --hostfile /home/.grade/HW4/4_hosts ../pi_nonblock_linear 1000000000
mpirun -np 16 --hostfile /home/.grade/HW4/4_hosts ../../ref/pi_nonblock_linear 1000000000