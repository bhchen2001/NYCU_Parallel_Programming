mpicxx ../hello.cc -o ../hello
./scp_script.sh
mpirun -np 5 --hostfile /home/.grade/HW4/2_hosts ../hello
mpirun -np 7 --hostfile /home/.grade/HW4/2_hosts ../hello
mpirun -np 10 --hostfile /home/.grade/HW4/3_hosts ../hello
mpirun -np 11 --hostfile /home/.grade/HW4/3_hosts ../hello
mpirun -np 16 --hostfile /home/.grade/HW4/4_hosts ../hello