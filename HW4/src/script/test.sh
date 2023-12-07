# run the following command for 100000 times using for loop
# mpirun -np 16 --hostfile /home/.grade/HW4/4_hosts pi_reduce 1000000000

mpicxx ../pi_nonblock_linear.cc -o ../pi_nonblock_linear
parallel-scp -h host.txt -r ~/HW4/ ~/

for i in {1..100}
do
    mpirun -np 5 --hostfile /home/.grade/HW4/2_hosts ../pi_nonblock_linear 1000000000 >> result_test
    mpirun -np 7 --hostfile /home/.grade/HW4/2_hosts ../pi_nonblock_linear 1000000000 >> result_test
    mpirun -np 10 --hostfile /home/.grade/HW4/3_hosts ../pi_nonblock_linear 1000000000 >> result_test
    mpirun -np 11 --hostfile /home/.grade/HW4/3_hosts ../pi_nonblock_linear 1000000000 >> result_test
    mpirun -np 16 --hostfile /home/.grade/HW4/4_hosts ../pi_nonblock_linear 1000000000 >> result_test
done