num_p=16

mpicxx ../pi_one_side.cc -o ../pi_one_side
./scp_script.sh
mpirun -np 5 --hostfile /home/.grade/HW4/2_hosts ../pi_one_side 1000000000
mpirun -np 5 --hostfile /home/.grade/HW4/2_hosts ../../ref/pi_one_side 1000000000
mpirun -np 7 --hostfile /home/.grade/HW4/2_hosts ../pi_one_side 1000000000
mpirun -np 7 --hostfile /home/.grade/HW4/2_hosts ../../ref/pi_one_side 1000000000
mpirun -np 10 --hostfile /home/.grade/HW4/3_hosts ../pi_one_side 1000000000
mpirun -np 10 --hostfile /home/.grade/HW4/3_hosts ../../ref/pi_one_side 1000000000
mpirun -np 11 --hostfile /home/.grade/HW4/3_hosts ../pi_one_side 1000000000
mpirun -np 11 --hostfile /home/.grade/HW4/3_hosts ../../ref/pi_one_side 1000000000
mpirun -np 16 --hostfile /home/.grade/HW4/4_hosts ../pi_one_side 1000000000
mpirun -np 16 --hostfile /home/.grade/HW4/4_hosts ../../ref/pi_one_side 1000000000