make clean && make
parallel-scp -h ./script/host.txt -r ~/HW4/ ~/
# mpirun -np 2 --hostfile mat1_hosts matmul < data-set/sample2
mpirun -np 4 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data1_1 > result_1
mpirun -np 4 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data1_2 >> result_1
mpirun -np 4 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data1_4 >> result_1
mpirun -np 4 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data1_4 >> result_1
mpirun -np 4 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data1_5 >> result_1
mpirun -np 4 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data1_6 >> result_1
mpirun -np 4 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data1_7 >> result_1
mpirun -np 4 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data1_8 >> result_1
mpirun -np 4 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data1_9 >> result_1
mpirun -np 4 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data1_10 >> result_1
mpirun -np 8 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data2_1 > result_2
mpirun -np 8 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data2_2 >> result_2
mpirun -np 8 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data2_3 >> result_2
mpirun -np 8 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data2_4 >> result_2
mpirun -np 8 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data2_5 >> result_2
mpirun -np 8 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data2_6 >> result_2
mpirun -np 8 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data2_7 >> result_2
mpirun -np 8 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data2_8 >> result_2
mpirun -np 8 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data2_9 >> result_2
mpirun -np 8 --hostfile mat9_hosts ./matmul < /home/.grade/HW4/data-set/data2_10 >> result_2